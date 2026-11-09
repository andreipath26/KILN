//! Expert blob reader with a byte-bounded expert cache and pinned leases.
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use kiln_mem::{ExpertCache, ExpertCachePolicy, ExpertCacheStats, ExpertKey};
use kiln_models::ExpertTable;

use crate::{ExpertBlobError, ExpertBlobReader};

const MAX_ACCESS_TRACE_EVENTS: usize = 1_000_000;

/// A bounded snapshot of successful expert access requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertAccessTraceSnapshot {
    pub events: Vec<ExpertKey>,
    pub max_events: usize,
    pub dropped_events: u64,
}

/// Opt-in model-keyed warm-up of the hottest verified experts.
/// Counters describe cache residency, not verified inference speedups.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertWarmupReport {
    pub requested: usize,
    pub admitted: usize,
    pub already_resident: usize,
    pub duplicate_keys: usize,
    pub skipped_budget: usize,
    pub rejected_pinned: usize,
    pub selected_bytes: u64,
    pub admitted_bytes: u64,
}

struct AccessTraceState {
    max_events: usize,
    events: Vec<ExpertKey>,
    dropped_events: u64,
}

/// Reads expert blocks from the blob and retains hot blocks under a strict
/// cache-membership byte budget. Concurrent misses are rechecked after taking
/// the reader lock so only one read of a missing key is needed at a time.
pub struct CachedExpertBlobReader {
    reader: Mutex<ExpertBlobReader>,
    cache: ExpertCache,
    access_trace: Mutex<AccessTraceState>,
}

impl CachedExpertBlobReader {
    pub fn open(
        path: impl AsRef<Path>,
        table: ExpertTable,
        cache_capacity_bytes: u64,
        policy: ExpertCachePolicy,
    ) -> Result<Self, ExpertBlobError> {
        let reader = ExpertBlobReader::open(path, table)?;
        Ok(Self::from_reader(reader, cache_capacity_bytes, policy))
    }

    /// Open a blob using a size-bounded JSON ExpertTable index file.
    pub fn open_indexed(
        path: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
        cache_capacity_bytes: u64,
        policy: ExpertCachePolicy,
    ) -> Result<Self, ExpertBlobError> {
        let reader = ExpertBlobReader::open_indexed(path, index_path)?;
        Ok(Self::from_reader(reader, cache_capacity_bytes, policy))
    }

    fn from_reader(
        reader: ExpertBlobReader,
        cache_capacity_bytes: u64,
        policy: ExpertCachePolicy,
    ) -> Self {
        Self {
            reader: Mutex::new(reader),
            cache: ExpertCache::new(cache_capacity_bytes, policy),
            access_trace: Mutex::new(AccessTraceState {
                max_events: 0,
                events: Vec::new(),
                dropped_events: 0,
            }),
        }
    }

    fn lock_reader(&self) -> MutexGuard<'_, ExpertBlobReader> {
        self.reader.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn record_access(&self, key: ExpertKey) {
        let mut trace = self.access_trace.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if trace.max_events == 0 {
            return;
        }
        if trace.events.len() < trace.max_events {
            trace.events.push(key);
        } else {
            trace.dropped_events = trace.dropped_events.saturating_add(1);
        }
    }

    /// Enable bounded successful-read tracing. A zero limit disables tracing.
    /// Existing events and the dropped-event counter are reset.
    pub fn enable_access_trace(&self, max_events: usize) {
        let mut trace = self.access_trace.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let max_events = max_events.min(MAX_ACCESS_TRACE_EVENTS);
        trace.max_events = max_events;
        trace.events = Vec::with_capacity(max_events);
        trace.dropped_events = 0;
    }

    /// Return a stable snapshot of the recorded (layer, expert_id) sequence.
    pub fn access_trace_snapshot(&self) -> ExpertAccessTraceSnapshot {
        let trace = self.access_trace.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        ExpertAccessTraceSnapshot {
            events: trace.events.clone(),
            max_events: trace.max_events,
            dropped_events: trace.dropped_events,
        }
    }

    /// Export a bounded, private router trace to a new file only. Model
    /// routing patterns may reveal workload characteristics; never overwrite
    /// another file or write them world-readable on Unix.
    pub fn write_access_trace_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        use std::io::Write;
        let snapshot = self.access_trace_snapshot();
        let bytes = serde_json::to_vec_pretty(&snapshot.events)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut output = options.open(path)?;
        if let Err(error) = output.write_all(&bytes).and_then(|_| output.sync_all()) {
            // Do not leave a truncated routing trace that might be ingested
            // as legitimate evidence by a later optimization pass.
            drop(output);
            let _ = std::fs::remove_file(path);
            return Err(error);
        }
        Ok(())
    }

    /// Backwards-compatible owned result; callers requiring a bounded
    /// expert working set should prefer read_expert_shared to avoid a
    /// second full-block copy for every cache hit.
    pub fn read_expert(
        &self,
        layer: u32,
        expert_id: u32,
    ) -> Result<Vec<u8>, ExpertBlobError> {
        Ok(self.read_expert_shared(layer, expert_id)?.as_ref().to_vec())
    }

    /// Return a reference-counted expert allocation. Hits do not duplicate
    /// the underlying block and concurrent cold requests are coalesced.
    /// Active Arc leases prevent eviction of the retained entry. A cold
    /// miss that cannot be cached may still return transient bytes; callers
    /// must separately bound their aggregate in-flight allocations.
    pub fn read_expert_shared(
        &self,
        layer: u32,
        expert_id: u32,
    ) -> Result<Arc<[u8]>, ExpertBlobError> {
        let key = ExpertKey { layer, expert_id };
        if let Some(bytes) = self.cache.get_shared_if_present(key) {
            self.record_access(key);
            return Ok(bytes);
        }

        let mut reader = self.lock_reader();
        // Avoid duplicate I/O when another thread filled the cache while
        // this caller waited for the single file reader lock.
        if let Some(bytes) = self.cache.get_shared_if_present(key) {
            self.record_access(key);
            return Ok(bytes);
        }

        self.cache.record_miss();
        let bytes: Arc<[u8]> = Arc::from(reader.read_expert(layer, expert_id)?);
        // Admission can reject a cold block while all existing entries
        // are pinned. Correctness takes precedence over cache residency.
        let _admission = self.cache.insert_shared(key, Arc::clone(&bytes));
        self.record_access(key);
        Ok(bytes)
    }

    /// Copy an expert into a caller-owned reusable buffer. This avoids
    /// allocating an owned Vec on warm cache hits, but external buffers
    /// must be included in the caller's memory budget.
    pub fn read_expert_into(
        &self,
        layer: u32,
        expert_id: u32,
        buffer: &mut [u8],
    ) -> Result<(), ExpertBlobError> {
        // Validate before a cold read or cache lookup: an undersized
        // destination must never cause avoidable SSD I/O or cache churn.
        let expected = self.lock_reader().expert_byte_length(layer, expert_id)?;
        if buffer.len() as u64 != expected {
            return Err(ExpertBlobError::BufferLengthMismatch {
                expected,
                actual: buffer.len(),
            });
        }
        let bytes = self.read_expert_shared(layer, expert_id)?;
        buffer.copy_from_slice(&bytes);
        Ok(())
    }

    /// Warm externally ranked expert keys with exact index sizes and a strict
    /// byte budget. This is a placement optimization, NEVER a prediction of
    /// which expert the actual model router will select. No tensors are
    /// activated unless the caller explicitly invokes this method.
    ///
    /// Admission may fail when existing read leases pin cache capacity.
    /// Hold newly warmed blocks until the batch is complete so lower-priority
    /// entries cannot displace previously admitted, hotter experts.
    pub fn prewarm_ranked(
        &self,
        ranked: &[ExpertKey],
        requested_budget_bytes: u64,
    ) -> Result<ExpertWarmupReport, ExpertBlobError> {
        const MAX_WARM_KEYS: usize = 65_536;
        if ranked.len() > MAX_WARM_KEYS {
            return Err(ExpertBlobError::InvalidRoute(
                "prewarm expert count exceeds the bounded request limit".into(),
            ));
        }
        let capacity = self.cache.stats().capacity_bytes;
        let budget = requested_budget_bytes.min(capacity);
        let mut result = ExpertWarmupReport {
            requested: ranked.len(),
            ..ExpertWarmupReport::default()
        };
        let mut unique = std::collections::HashSet::new();
        let mut plan = Vec::new();
        let mut planned_bytes = 0u64;
        // Validate *all* keys before starting any read so a malformed
        // model-keyed history cannot cause partial disk I/O.
        for &key in ranked {
            if !unique.insert(key) {
                result.duplicate_keys += 1;
                continue;
            }
            let bytes = self.expert_byte_length(key.layer, key.expert_id)?;
            if bytes > budget.saturating_sub(planned_bytes) {
                result.skipped_budget += 1;
                continue;
            }
            planned_bytes += bytes;
            plan.push((key, bytes));
        }
        result.selected_bytes = planned_bytes;
        let mut pinned: Vec<Arc<[u8]>> = Vec::with_capacity(plan.len());
        for (key, bytes) in plan {
            let was_cached = self.cache.peek_shared(key);
            if let Some(handle) = was_cached {
                result.already_resident += 1;
                result.admitted += 1;
                result.admitted_bytes += bytes;
                pinned.push(handle);
                continue;
            }
            let expert = self.read_expert_shared(key.layer, key.expert_id)?;
            if self.cache.peek_shared(key).is_some() {
                result.admitted += 1;
                result.admitted_bytes += bytes;
                pinned.push(expert);
            } else {
                result.rejected_pinned += 1;
            }
        }
        Ok(result)
    }

    pub fn invalidate(&self, layer: u32, expert_id: u32) -> bool {
        self.cache.invalidate(ExpertKey { layer, expert_id })
    }

    pub fn clear_cache(&self) -> usize {
        self.cache.clear()
    }

    pub fn cache_stats(&self) -> ExpertCacheStats {
        self.cache.stats()
    }

    /// Index-only expert size lookup; does not fetch expert payload.
    pub fn expert_byte_length(&self, layer: u32, expert_id: u32)
        -> Result<u64, ExpertBlobError> {
        self.lock_reader().expert_byte_length(layer, expert_id)
    }

    pub fn file_size(&self) -> u64 {
        self.lock_reader().file_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::ExpertRange;
    use std::io::Write;
    use std::sync::Arc;
    use std::thread;

    fn fixture() -> (std::path::PathBuf, Vec<u8>, ExpertTable) {
        let path = std::env::temp_dir().join(format!(
            "kiln-cached-expert-blob-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes: Vec<u8> = (0..32).collect();
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        let table = ExpertTable {
            blob_version: 1,
            blob_size: bytes.len() as u64,
            ranges: vec![
                ExpertRange { layer: 0, expert_id: 0, byte_offset: 0, byte_length: 8 },
                ExpertRange { layer: 0, expert_id: 1, byte_offset: 8, byte_length: 8 },
                ExpertRange { layer: 1, expert_id: 0, byte_offset: 16, byte_length: 8 },
                ExpertRange { layer: 1, expert_id: 1, byte_offset: 24, byte_length: 8 },
            ],
        };
        (path, bytes, table)
    }

    #[test]
    fn cold_then_warm_read_uses_cache_and_accounts_bytes() {
        let (path, bytes, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 8, ExpertCachePolicy::Lru).unwrap();
        assert_eq!(reader.read_expert(0, 1).unwrap(), bytes[8..16]);
        assert_eq!(reader.read_expert(0, 1).unwrap(), bytes[8..16]);
        let stats = reader.cache_stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.used_bytes, 8);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn shared_expert_cache_hits_return_same_backing_allocation() {
        let (path, bytes, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 8, ExpertCachePolicy::Lru).unwrap();
        let first = reader.read_expert_shared(0, 1).unwrap();
        let second = reader.read_expert_shared(0, 1).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref(), &bytes[8..16]);
        assert_eq!(reader.cache_stats().hits, 1);
        assert_eq!(reader.cache_stats().misses, 1);
        assert_eq!(reader.cache_stats().used_bytes, 8);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reuse_caller_buffer_without_allocating_owned_result() {
        let (path, bytes, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 8, ExpertCachePolicy::Lru).unwrap();
        let mut buffer = [0u8; 8];
        assert!(matches!(
            reader.read_expert_into(0, 1, &mut [0u8; 4]),
            Err(ExpertBlobError::BufferLengthMismatch { expected: 8, actual: 4 })
        ));
        assert_eq!(reader.cache_stats().hits, 0);
        assert_eq!(reader.cache_stats().misses, 0);
        reader.read_expert_into(0, 1, &mut buffer).unwrap();
        assert_eq!(buffer.as_slice(), &bytes[8..16]);
        buffer.fill(0);
        reader.read_expert_into(0, 1, &mut buffer).unwrap();
        assert_eq!(buffer.as_slice(), &bytes[8..16]);
        assert_eq!(reader.cache_stats().hits, 1);
        assert_eq!(reader.cache_stats().misses, 1);
        assert!(matches!(
            reader.read_expert_into(0, 1, &mut [0u8; 4]),
            Err(ExpertBlobError::BufferLengthMismatch { expected: 8, actual: 4 })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn external_shared_reader_outlives_eviction_without_use_after_free() {
        let (path, bytes, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 8, ExpertCachePolicy::Lru).unwrap();
        let held = reader.read_expert_shared(0, 0).unwrap();
        let next = reader.read_expert_shared(1, 0).unwrap();
        assert_eq!(next.as_ref(), &bytes[16..24]);
        assert_eq!(held.as_ref(), &bytes[0..8]);
        assert_eq!(reader.cache_stats().used_bytes, 8);
        assert!(!Arc::ptr_eq(&held, &next));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn oversized_expert_is_returned_but_not_cached() {
        let (path, bytes, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 4, ExpertCachePolicy::Lru).unwrap();
        assert_eq!(reader.read_expert(0, 0).unwrap(), bytes[0..8]);
        assert_eq!(reader.read_expert(0, 0).unwrap(), bytes[0..8]);
        assert_eq!(reader.cache_stats().used_bytes, 0);
        assert_eq!(reader.cache_stats().misses, 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn concurrent_same_key_reads_return_correct_bytes_and_stay_bounded() {
        let (path, bytes, table) = fixture();
        let reader = Arc::new(CachedExpertBlobReader::open(&path, table, 16, ExpertCachePolicy::Lru).unwrap());
        let expected = bytes[16..24].to_vec();
        let mut workers = Vec::new();
        for _ in 0..8 {
            let reader = Arc::clone(&reader);
            let expected = expected.clone();
            workers.push(thread::spawn(move || {
                assert_eq!(reader.read_expert(1, 0).unwrap(), expected);
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let stats = reader.cache_stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.used_bytes, 8);
        assert_eq!(stats.hits, 7);
        assert_eq!(stats.misses, 1);
        assert!(stats.used_bytes <= stats.capacity_bytes);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn concurrent_shared_misses_coalesce_to_one_block_allocation() {
        let (path, bytes, table) = fixture();
        let reader = Arc::new(CachedExpertBlobReader::open(
            &path, table, 16, ExpertCachePolicy::Lru
        ).unwrap());
        let mut workers = Vec::new();
        for _ in 0..8 {
            let reader = Arc::clone(&reader);
            workers.push(thread::spawn(move || reader.read_expert_shared(1, 0).unwrap()));
        }
        let handles: Vec<_> = workers.into_iter().map(|worker| worker.join().unwrap()).collect();
        for handle in &handles {
            assert!(Arc::ptr_eq(handle, &handles[0]));
            assert_eq!(handle.as_ref(), &bytes[16..24]);
        }
        let stats = reader.cache_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 7);
        assert_eq!(stats.used_bytes, 8);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn opens_index_file_directly_and_uses_the_same_cache_path() {
        let (path, bytes, table) = fixture();
        let index_path = std::env::temp_dir().join(format!(
            "kiln-cached-index-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::write(&index_path, serde_json::to_vec(&table).unwrap()).unwrap();
        let reader = CachedExpertBlobReader::open_indexed(
            &path, &index_path, 16, ExpertCachePolicy::Lru
        ).unwrap();
        assert_eq!(reader.read_expert(1, 0).unwrap(), bytes[16..24]);
        assert_eq!(reader.read_expert(1, 0).unwrap(), bytes[16..24]);
        assert_eq!(reader.cache_stats().hits, 1);
        assert_eq!(reader.cache_stats().misses, 1);
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_access_trace_records_successful_reads_and_exports_json() {
        let (path, _, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 16, ExpertCachePolicy::Lru).unwrap();
        reader.enable_access_trace(2);
        reader.read_expert(0, 0).unwrap();
        reader.read_expert(1, 0).unwrap();
        reader.read_expert(0, 0).unwrap();
        let snapshot = reader.access_trace_snapshot();
        assert_eq!(snapshot.events, vec![
            ExpertKey { layer: 0, expert_id: 0 },
            ExpertKey { layer: 1, expert_id: 0 },
        ]);
        assert_eq!(snapshot.max_events, 2);
        assert_eq!(snapshot.dropped_events, 1);

        let trace_path = std::env::temp_dir().join(format!(
            "kiln-access-trace-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        reader.write_access_trace_json(&trace_path).unwrap();
        // Repeated exports must fail rather than silently overwrite evidence.
        assert_eq!(reader.write_access_trace_json(&trace_path).unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&trace_path).unwrap()
                .permissions().mode() & 0o777, 0o600);
        }
        let exported: Vec<ExpertKey> = serde_json::from_slice(&std::fs::read(&trace_path).unwrap()).unwrap();
        assert_eq!(exported, snapshot.events);
        reader.enable_access_trace(0);
        reader.read_expert(0, 1).unwrap();
        assert!(reader.access_trace_snapshot().events.is_empty());
        assert_eq!(reader.access_trace_snapshot().max_events, 0);
        assert_eq!(reader.access_trace_snapshot().dropped_events, 0);
        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prewarm_ranked_deduplicates_and_respects_strict_cache_budget() {
        let (path, data, table) = fixture();
        let reader = CachedExpertBlobReader::open(
            &path, table, 16, ExpertCachePolicy::Lru
        ).unwrap();
        let a = ExpertKey { layer: 0, expert_id: 0 };
        let b = ExpertKey { layer: 0, expert_id: 1 };
        let c = ExpertKey { layer: 1, expert_id: 0 };
        let result = reader.prewarm_ranked(&[a, a, b, c], 16).unwrap();
        assert_eq!(result.requested, 4);
        assert_eq!(result.duplicate_keys, 1);
        assert_eq!(result.skipped_budget, 1);
        assert_eq!(result.admitted, 2);
        assert_eq!(result.selected_bytes, 16);
        assert_eq!(result.admitted_bytes, 16);
        // Passive residency inspection during prewarm is not a real cache hit.
        assert_eq!(reader.cache_stats().hits, 0);
        assert_eq!(reader.cache_stats().used_bytes, 16);
        assert_eq!(reader.read_expert(0, 0).unwrap(), data[0..8]);
        assert_eq!(reader.read_expert(0, 1).unwrap(), data[8..16]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prewarm_rejects_bad_key_before_any_disk_read() {
        let (path, _, table) = fixture();
        let reader = CachedExpertBlobReader::open(
            &path, table, 16, ExpertCachePolicy::Lru
        ).unwrap();
        let good = ExpertKey { layer: 0, expert_id: 0 };
        let bad = ExpertKey { layer: 99, expert_id: 0 };
        assert!(reader.prewarm_ranked(&[good, bad], 16).is_err());
        assert_eq!(reader.cache_stats().misses, 0);
        assert_eq!(reader.cache_stats().entries, 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prewarm_does_not_evict_expert_held_by_computation() {
        let (path, _, table) = fixture();
        let reader = CachedExpertBlobReader::open(
            &path, table, 8, ExpertCachePolicy::Adaptive
        ).unwrap();
        let held = reader.read_expert_shared(0, 0).unwrap();
        let other = ExpertKey { layer: 0, expert_id: 1 };
        let refused = reader.prewarm_ranked(&[other], 8).unwrap();
        assert_eq!(refused.rejected_pinned, 1);
        assert_eq!(reader.cache_stats().used_bytes, 8);
        assert_eq!(reader.cache_stats().entries, 1);
        drop(held);
        let accepted = reader.prewarm_ranked(&[other], 8).unwrap();
        assert_eq!(accepted.admitted, 1);
        assert_eq!(reader.cache_stats().used_bytes, 8);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalidation_releases_cache_bytes() {
        let (path, _, table) = fixture();
        let reader = CachedExpertBlobReader::open(&path, table, 16, ExpertCachePolicy::Lru).unwrap();
        reader.read_expert(0, 0).unwrap();
        assert!(reader.invalidate(0, 0));
        assert_eq!(reader.cache_stats().used_bytes, 0);
        assert_eq!(reader.clear_cache(), 0);
        std::fs::remove_file(path).unwrap();
    }
}
