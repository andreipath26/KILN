//! Phase 5.3: bounded portable asynchronous expert-read fallback.
//!
//! This worker overlaps buffered source reads with caller computation.
//! It does not claim io_uring/SQPOLL or hide a miss on the target Dell.
//! All returned buffers are shared and need explicit caller memory accounting.
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}, mpsc};
use std::thread::{self, JoinHandle};
use kiln_mem::ExpertKey;
use crate::CachedExpertBlobReader;
#[cfg(target_os = "linux")]
use crate::LinuxUringExpertReader;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsyncExpertStats {
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub skipped_stale: u64,
    pub rejected_capacity: u64,
    pub pending_requests: usize,
    pub pending_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitResult { Submitted, AlreadyPending, QueueFull, ByteLimit, StaleEpoch, MissingKey }

pub struct ExpertReadCompletion {
    pub epoch: u64,
    pub key: ExpertKey,
    pub result: Result<Arc<[u8]>, String>,
}
struct ReadRequest { epoch: u64, key: ExpertKey }
enum WorkerReadBackend {
    Cached(Arc<CachedExpertBlobReader>),
    #[cfg(target_os = "linux")]
    Uring(LinuxUringExpertReader),
}
impl WorkerReadBackend {
    fn read(&mut self, key: ExpertKey) -> Result<Arc<[u8]>, String> {
        match self {
            Self::Cached(reader) => reader.read_expert_shared(key.layer,key.expert_id)
                .map_err(|e|e.to_string()),
            #[cfg(target_os = "linux")]
            Self::Uring(reader) => reader.read_expert(key.layer,key.expert_id)
                .map(Arc::from).map_err(|e|e.to_string()),
        }
    }
}
struct WorkerState {
    recv: mpsc::Receiver<ReadRequest>,
    sender: mpsc::Sender<ExpertReadCompletion>,
    backend: WorkerReadBackend,
    epoch: Arc<AtomicU64>,
}

/// One dedicated worker and one bounded request channel. Pending byte count
/// includes queued, running, and undrained completed tasks. An externally
/// retained completion Arc remains outside the pending byte budget.
pub struct AsyncExpertPrefetch {
    sender: Option<mpsc::SyncSender<ReadRequest>>,
    done: mpsc::Receiver<ExpertReadCompletion>,
    worker: Option<JoinHandle<()>>,
    reader: Arc<CachedExpertBlobReader>,
    epoch: Arc<AtomicU64>,
    // Include the epoch in identity. A stale completion must not release
    // a newer request for the same expert or mask its admission.
    pending: HashMap<(u64, ExpertKey), u64>,
    max_requests: usize,
    max_bytes: u64,
    stats: AsyncExpertStats,
}

impl AsyncExpertPrefetch {
    pub fn new(
        reader: Arc<CachedExpertBlobReader>,
        max_requests: usize,
        max_bytes: u64,
    ) -> Self {
        Self::new_with_backend(
            Arc::clone(&reader), WorkerReadBackend::Cached(reader),
            max_requests, max_bytes,
        )
    }

    /// Used only when the async and authoritative readers were both opened
    /// by the owning coordinator from the same validated expert source.
    #[cfg(target_os = "linux")]
    pub(crate) fn new_with_uring_reader(
        reader: Arc<CachedExpertBlobReader>,
        uring: LinuxUringExpertReader,
        max_requests: usize,
        max_bytes: u64,
    ) -> Self {
        Self::new_with_backend(reader, WorkerReadBackend::Uring(uring),
            max_requests, max_bytes)
    }

    fn new_with_backend(
        reader: Arc<CachedExpertBlobReader>,
        backend: WorkerReadBackend,
        max_requests: usize,
        max_bytes: u64,
    ) -> Self {
        let queue_capacity = max_requests.max(1);
        let (sender, recv) = mpsc::sync_channel(queue_capacity);
        let (completed, done) = mpsc::channel();
        let epoch = Arc::new(AtomicU64::new(0));
        let mut worker_state = WorkerState {
            recv, sender: completed, backend, epoch: Arc::clone(&epoch),
        };
        let worker = thread::spawn(move || {
            while let Ok(request) = worker_state.recv.recv() {
                if worker_state.epoch.load(Ordering::Acquire) != request.epoch {
                    let _ = worker_state.sender.send(ExpertReadCompletion {
                        epoch:request.epoch,key:request.key,
                        result:Err("stale epoch; read skipped".to_owned()),
                    });
                    continue;
                }
                let result=worker_state.backend.read(request.key);
                let _ = worker_state.sender.send(ExpertReadCompletion {
                    epoch:request.epoch,key:request.key,result,
                });
            }
        });
        Self {
            sender:Some(sender),done,worker:Some(worker),reader,epoch,
            pending:HashMap::new(),max_requests,max_bytes,
            stats:AsyncExpertStats::default(),
        }
    }

    /// Increment the active token epoch; queued stale jobs are skipped.
    /// Already running reads can still complete, but their output is rejected
    /// by the epoch check in poll_ready.
    pub fn begin_epoch(&mut self, epoch: u64) -> Result<(), &'static str> {
        if epoch <= self.epoch.load(Ordering::Acquire) {
            return Err("epoch must strictly increase");
        }
        self.epoch.store(epoch, Ordering::Release);
        Ok(())
    }

    /// Reserve bytes before enqueuing, so invalid/oversized predictions
    /// cannot initiate I/O. The index lookup does not read expert payload.
    pub fn submit(&mut self, epoch: u64, key: ExpertKey) -> SubmitResult {
        if epoch == 0 || self.epoch.load(Ordering::Acquire) != epoch {
            return SubmitResult::StaleEpoch;
        }
        if self.pending.contains_key(&(epoch, key)) { return SubmitResult::AlreadyPending; }
        if self.pending.len() >= self.max_requests {
            self.stats.rejected_capacity += 1;
            return SubmitResult::QueueFull;
        }
        let size = match self.reader.expert_byte_length(key.layer,key.expert_id) {
            Ok(size) if size > 0 => size,
            _ => return SubmitResult::MissingKey,
        };
        if size > self.max_bytes.saturating_sub(self.stats.pending_bytes) {
            self.stats.rejected_capacity += 1;
            return SubmitResult::ByteLimit;
        }
        let Some(sender) = &self.sender else { return SubmitResult::QueueFull };
        if sender.try_send(ReadRequest { epoch, key }).is_err() {
            self.stats.rejected_capacity += 1;
            return SubmitResult::QueueFull;
        }
        self.pending.insert((epoch,key),size);
        self.stats.pending_bytes += size;
        self.stats.pending_requests = self.pending.len();
        self.stats.submitted += 1;
        SubmitResult::Submitted
    }

    /// Nonblocking completion drain. A stale result is omitted rather than
    /// exposed for a new router epoch. The returned Arc handle is now owned
    /// by the caller; the pending budget is released.
    pub fn poll_ready(&mut self) -> Vec<ExpertReadCompletion> {
        let mut results=Vec::new();
        while let Ok(item) = self.done.try_recv() {
            if let Some(size)=self.pending.remove(&(item.epoch,item.key)) {
                self.stats.pending_bytes=self.stats.pending_bytes.saturating_sub(size);
            }
            self.stats.completed+=1;
            if self.epoch.load(Ordering::Acquire) != item.epoch {
                self.stats.skipped_stale+=1;
            } else {
                if item.result.is_err() { self.stats.failed+=1; }
                results.push(item);
            }
        }
        self.stats.pending_requests=self.pending.len();
        results
    }

    pub fn stats(&self)->AsyncExpertStats { self.stats }
}

impl Drop for AsyncExpertPrefetch {
    fn drop(&mut self) {
        // Disconnect sender. Worker drains already queued tasks then exits;
        // a running file read is not preempted.
        self.sender.take();
        if let Some(worker)=self.worker.take() { let _=worker.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::{ExpertTable, ExpertRange};
    fn key(id:u32)->ExpertKey{ExpertKey{layer:0,expert_id:id}}

    #[test]
    fn asynchronously_reads_exact_experts_without_exceeding_admission_caps() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("data.bin");
        let bytes:Vec<u8>=(0..64u8).collect();
        std::fs::write(&path,&bytes).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:64,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
            ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
        ]};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &path,table,32,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut prefetch=AsyncExpertPrefetch::new(reader,1,16);
        prefetch.begin_epoch(1).unwrap();
        assert_eq!(prefetch.submit(1,key(0)),SubmitResult::Submitted);
        assert_eq!(prefetch.submit(1,key(0)),SubmitResult::AlreadyPending);
        assert_eq!(prefetch.submit(1,key(1)),SubmitResult::QueueFull);
        assert_eq!(prefetch.stats().pending_bytes,16);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        let completion=loop {
            let mut ready=prefetch.poll_ready();
            if !ready.is_empty(){break ready.remove(0);}
            assert!(std::time::Instant::now()<deadline,"asynchronous expert read timed out");
            std::thread::yield_now();
        };
        assert_eq!(completion.key,key(0));
        assert_eq!(completion.result.unwrap().as_ref(),&bytes[0..16]);
        assert_eq!(prefetch.stats().pending_bytes,0);
        assert_eq!(prefetch.submit(1,key(99)),SubmitResult::MissingKey);
        assert_eq!(prefetch.submit(2,key(1)),SubmitResult::StaleEpoch);
        drop(prefetch);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn stale_completion_cannot_erase_new_epoch_same_expert_reservation() {
        let root = std::env::temp_dir().join(format!(
            "kiln-epoch-overlap-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("data.bin");
        std::fs::write(&path, [5u8; 16]).unwrap();
        let table = ExpertTable {blob_version:1,blob_size:16,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
        ]};
        let reader = Arc::new(CachedExpertBlobReader::open(
            &path, table, 16, ExpertCachePolicy::Lru
        ).unwrap());
        let mut prefetch = AsyncExpertPrefetch::new(reader, 2, 32);
        prefetch.begin_epoch(1).unwrap();
        assert_eq!(prefetch.submit(1, key(0)), SubmitResult::Submitted);
        prefetch.begin_epoch(2).unwrap();
        // The same key was queued in an older epoch, but must still be
        // admissible for the current router decision.
        assert_eq!(prefetch.submit(2, key(0)), SubmitResult::Submitted);
        assert_eq!(prefetch.stats().pending_requests, 2);
        assert_eq!(prefetch.stats().pending_bytes, 32);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        let mut current = Vec::new();
        while prefetch.stats().completed < 2 {
            current.extend(prefetch.poll_ready());
            assert!(std::time::Instant::now() < deadline,
                "stale/current completion pair timed out");
            std::thread::yield_now();
        }
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].epoch, 2);
        assert_eq!(current[0].result.as_ref().unwrap().as_ref(), &[5u8;16]);
        assert_eq!(prefetch.stats().pending_requests, 0);
        assert_eq!(prefetch.stats().pending_bytes, 0);
        assert_eq!(prefetch.stats().skipped_stale, 1);
        drop(prefetch);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn rejects_byte_cap_without_starting_expert_read() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-reject-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("data.bin");
        std::fs::write(&path,[9u8;16]).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:16,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
        ]};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &path,table,16,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut prefetch=AsyncExpertPrefetch::new(Arc::clone(&reader),2,8);
        prefetch.begin_epoch(1).unwrap();
        assert_eq!(prefetch.submit(1,key(0)),SubmitResult::ByteLimit);
        assert_eq!(reader.cache_stats().misses,0);
        drop(prefetch);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
