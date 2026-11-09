//! Phase 5.3: bounded portable asynchronous expert-read fallback.
//!
//! This worker overlaps buffered source reads with caller computation.
//! It does not claim io_uring/SQPOLL or hide a miss on the target Dell.
//! All returned buffers are shared and need explicit caller memory accounting.
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}, mpsc};
use std::thread::{self, JoinHandle};
use kiln_mem::ExpertKey;
use crate::{CachedExpertBlobReader, ExpertBlobError};
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
    /// Number of independent read workers, not a measured I/O speedup.
    pub worker_count: usize,
    /// Elapsed worker service time including cache lookup / read. This is
    /// NOT physical NVMe latency and does not prove I/O/compute overlap.
    pub worker_service_nanos: u64,
    /// Bytes returned by completed successful predictions, including cached.
    pub completed_operand_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitResult { Submitted, AlreadyPending, QueueFull, ByteLimit, StaleEpoch, MissingKey }

pub struct ExpertReadCompletion {
    pub epoch: u64,
    pub key: ExpertKey,
    pub result: Result<Arc<[u8]>, String>,
    pub worker_service_nanos: u64,
}
struct ReadRequest { epoch: u64, key: ExpertKey }
enum WorkerReadBackend {
    Cached(Arc<CachedExpertBlobReader>),
    Forked {
        cache: Arc<CachedExpertBlobReader>,
        source: crate::ExpertBlobReader,
    },
    #[cfg(target_os = "linux")]
    Uring(LinuxUringExpertReader),
}
impl WorkerReadBackend {
    fn read(&mut self, key: ExpertKey) -> Result<Arc<[u8]>, String> {
        match self {
            Self::Cached(reader) => reader.read_speculative_expert_shared(key)
                .map_err(|e|e.to_string()),
            Self::Forked { cache, source } => cache.read_speculative_with_fork(source, key)
                .map_err(|e| e.to_string()),
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
    workers: Vec<JoinHandle<()>>,
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

    /// Launch a bounded portable worker pool with independently forked file
    /// descriptors. All workers share the same verified source index and one
    /// admission/cache budget, but can execute positional reads concurrently.
    /// Unlike a single mutex-backed cached reader, this can overlap SSD I/O.
    /// Measured throughput still depends on the hardware and page cache.
    pub fn new_parallel(
        reader: Arc<CachedExpertBlobReader>,
        worker_count: usize,
        max_requests: usize,
        max_bytes: u64,
    ) -> Result<Self, String> {
        if !(1..=8).contains(&worker_count) || max_requests == 0 || max_requests > 4096 {
            return Err("invalid bounded prefetch worker/request count".into());
        }
        // Prepare ALL independent file handles before starting any workers.
        let mut readers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            readers.push(reader.fork_source_reader().map_err(|e| e.to_string())?);
        }
        let (sender, recv) = mpsc::sync_channel::<ReadRequest>(max_requests);
        let recv = Arc::new(std::sync::Mutex::new(recv));
        let (completed, done) = mpsc::channel();
        let epoch = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(worker_count);
        for source in readers {
            let recv = Arc::clone(&recv);
            let sender_done = completed.clone();
            let current_epoch = Arc::clone(&epoch);
            let mut backend = WorkerReadBackend::Forked {
                cache: Arc::clone(&reader), source,
            };
            workers.push(thread::spawn(move || {
                loop {
                    let next = recv.lock().unwrap_or_else(|p| p.into_inner()).recv();
                    let Ok(request) = next else { break };
                    if current_epoch.load(Ordering::Acquire) != request.epoch {
                        let _ = sender_done.send(ExpertReadCompletion {
                            epoch: request.epoch, key: request.key,
                            result: Err("stale epoch; read skipped".into()),
                            worker_service_nanos: 0,
                        });
                        continue;
                    }
                    let started = std::time::Instant::now();
                    let result = backend.read(request.key);
                    let nanos = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    let _ = sender_done.send(ExpertReadCompletion {
                        epoch: request.epoch, key: request.key,
                        result, worker_service_nanos: nanos,
                    });
                }
            }));
        }
        drop(completed);
        Ok(Self {
            sender: Some(sender), done, workers, reader, epoch,
            pending: HashMap::new(), max_requests, max_bytes,
            stats: AsyncExpertStats {
                worker_count, ..AsyncExpertStats::default()
            },
        })
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
                        worker_service_nanos: 0,
                    });
                    continue;
                }
                let started = std::time::Instant::now();
                let result=worker_state.backend.read(request.key);
                let worker_service_nanos = started.elapsed().as_nanos()
                    .min(u64::MAX as u128) as u64;
                let _ = worker_state.sender.send(ExpertReadCompletion {
                    epoch:request.epoch,key:request.key,result,worker_service_nanos,
                });
            }
        });
        Self {
            sender:Some(sender),done,workers:vec![worker],reader,epoch,
            pending:HashMap::new(),max_requests,max_bytes,
            stats:AsyncExpertStats {
                worker_count: 1, ..AsyncExpertStats::default()
            },
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

    /// Submit an entire bounded batch in storage order, using the verified
    /// index to deduplicate and validate before scheduling any reads.
    /// A duplicate is reported as rejected but never occupies another slot.
    /// The actual model router still controls the order of committed experts.
    pub fn submit_batch(
        &mut self, epoch: u64, keys: &[ExpertKey],
    ) -> Result<(usize, usize), ExpertBlobError> {
        let ordered = self.reader.order_experts_by_source(keys)?;
        let duplicates = keys.len() - ordered.len();
        let mut submitted = 0usize;
        let mut rejected = duplicates;
        for key in ordered {
            if self.submit(epoch, key) == SubmitResult::Submitted {
                submitted += 1;
            } else {
                rejected += 1;
            }
        }
        Ok((submitted, rejected))
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
            self.stats.worker_service_nanos = self.stats.worker_service_nanos
                .saturating_add(item.worker_service_nanos);
            if let Ok(ref bytes) = item.result {
                self.stats.completed_operand_bytes = self.stats.completed_operand_bytes
                    .saturating_add(bytes.len() as u64);
            }
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
        for worker in self.workers.drain(..) { let _=worker.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::{ExpertTable, ExpertRange};
    fn key(id:u32)->ExpertKey{ExpertKey{layer:0,expert_id:id}}

    #[test]
    fn prefetch_batch_is_deduplicated_and_does_not_read_bad_keys() {
        let root=std::env::temp_dir().join(format!(
            "kiln-source-order-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("experts.bin");
        let bytes:Vec<u8>=(0..64u8).collect();
        std::fs::write(&path,&bytes).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:64,ranges:(0..4).map(|id| {
            ExpertRange{layer:0,expert_id:id,byte_offset:id as u64*16,byte_length:16}
        }).collect()};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &path,table,64,ExpertCachePolicy::Lru
        ).unwrap());
        let mut io=AsyncExpertPrefetch::new_parallel(
            Arc::clone(&reader), 2, 4, 64
        ).unwrap();
        let key=|id|ExpertKey{layer:0,expert_id:id};
        io.begin_epoch(1).unwrap();
        assert!(io.submit_batch(1,&[key(3),key(99)]).is_err());
        assert_eq!(reader.cache_stats().misses,0);
        assert_eq!(io.stats().pending_requests,0);
        assert_eq!(io.submit_batch(1,&[key(3),key(0),key(3),key(2)]).unwrap(),(3,1));
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        let mut complete=Vec::new();
        while complete.len()<3 {
            complete.extend(io.poll_ready());
            assert!(std::time::Instant::now()<deadline,"batch prefetch stalled");
            std::thread::yield_now();
        }
        let got:std::collections::HashSet<_>=complete.iter().map(|r| r.key).collect();
        assert_eq!(got,std::collections::HashSet::from([key(0),key(2),key(3)]));
        assert_eq!(io.stats().pending_bytes,0);
        drop(io);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn parallel_pool_reads_four_distinct_experts_without_losing_budget() {
        let root=std::env::temp_dir().join(format!(
            "kiln-parallel-worker-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("experts.bin");
        let data:Vec<u8>=(0..64u8).collect();
        std::fs::write(&path, &data).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:64,ranges:(0..4).map(|id| {
            ExpertRange{layer:0,expert_id:id,byte_offset:id as u64*16,byte_length:16}
        }).collect()};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &path,table,32,ExpertCachePolicy::Adaptive
        ).unwrap());
        assert!(AsyncExpertPrefetch::new_parallel(Arc::clone(&reader), 9, 4, 64).is_err());
        let mut io=AsyncExpertPrefetch::new_parallel(
            Arc::clone(&reader), 4, 4, 64
        ).unwrap();
        assert_eq!(io.stats().worker_count, 4);
        io.begin_epoch(1).unwrap();
        for id in 0..4 {
            assert_eq!(io.submit(1,key(id)),SubmitResult::Submitted);
        }
        assert_eq!(io.submit(1,key(0)),SubmitResult::AlreadyPending);
        assert_eq!(io.stats().pending_bytes,64);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        let mut done=Vec::new();
        while done.len()<4 {
            done.extend(io.poll_ready());
            assert!(std::time::Instant::now()<deadline,"parallel reads timed out");
            std::thread::yield_now();
        }
        done.sort_by_key(|x| x.key.expert_id);
        for (id,item) in done.into_iter().enumerate() {
            assert_eq!(item.key,key(id as u32));
            assert_eq!(item.result.unwrap().as_ref(),&data[id*16..id*16+16]);
        }
        assert_eq!(io.stats().pending_requests,0);
        assert_eq!(io.stats().pending_bytes,0);
        assert_eq!(io.stats().completed_operand_bytes,64);
        assert!(reader.cache_stats().used_bytes<=32);
        drop(io);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

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
        assert_eq!(prefetch.stats().completed_operand_bytes,16);
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
