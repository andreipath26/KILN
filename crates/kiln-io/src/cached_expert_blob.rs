//! Expert blob reader with a byte-bounded expert cache.
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use kiln_mem::{ExpertCache, ExpertCachePolicy, ExpertCacheStats, ExpertKey};
use kiln_models::ExpertTable;

use crate::{ExpertBlobError, ExpertBlobReader};

/// A bounded snapshot of successful expert access requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertAccessTraceSnapshot {
    pub events: Vec<ExpertKey>,
    pub dropped_events: u64,
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
        Ok(Self {
            reader: Mutex::new(ExpertBlobReader::open(path, table)?),
            cache: ExpertCache::new(cache_capacity_bytes, policy),
            access_trace: Mutex::new(AccessTraceState {
                max_events: 0,
                events: Vec::new(),
                dropped_events: 0,
            }),
        })
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
        trace.max_events = max_events;
        trace.events.clear();
        let additional = max_events.saturating_sub(trace.events.capacity());
        trace.events.reserve(additional);
        trace.dropped_events = 0;
    }

    /// Return a stable snapshot of the recorded (layer, expert_id) sequence.
    pub fn access_trace_snapshot(&self) -> ExpertAccessTraceSnapshot {
        let trace = self.access_trace.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        ExpertAccessTraceSnapshot {
            events: trace.events.clone(),
            dropped_events: trace.dropped_events,
        }
    }

    /// Write the recorded trace in the JSON format accepted by kiln-bench.
    pub fn write_access_trace_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let snapshot = self.access_trace_snapshot();
        let bytes = serde_json::to_vec_pretty(&snapshot.events)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        std::fs::write(path, bytes)
    }

    pub fn read_expert(
        &self,
        layer: u32,
        expert_id: u32,
    ) -> Result<Vec<u8>, ExpertBlobError> {
        let key = ExpertKey { layer, expert_id };
        if let Some(bytes) = self.cache.get(key) {
            self.record_access(key);
            return Ok(bytes);
        }

        let mut reader = self.lock_reader();
        // Another caller may have filled the cache while this caller waited
        // for the single file reader. Avoid duplicate disk reads in that case.
        if let Some(bytes) = self.cache.get_if_present(key) {
            self.record_access(key);
            return Ok(bytes);
        }

        let bytes = reader.read_expert(layer, expert_id)?;
        self.cache.insert(key, bytes.clone());
        self.record_access(key);
        Ok(bytes)
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
        assert_eq!(stats.hits + stats.misses, 8);
        assert!(stats.used_bytes <= stats.capacity_bytes);
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
        assert_eq!(snapshot.dropped_events, 1);

        let trace_path = std::env::temp_dir().join(format!(
            "kiln-access-trace-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        reader.write_access_trace_json(&trace_path).unwrap();
        let exported: Vec<ExpertKey> = serde_json::from_slice(&std::fs::read(&trace_path).unwrap()).unwrap();
        assert_eq!(exported, snapshot.events);
        reader.enable_access_trace(0);
        reader.read_expert(0, 1).unwrap();
        assert!(reader.access_trace_snapshot().events.is_empty());
        assert_eq!(reader.access_trace_snapshot().dropped_events, 0);
        std::fs::remove_file(trace_path).unwrap();
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
