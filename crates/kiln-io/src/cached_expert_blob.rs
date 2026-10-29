//! Expert blob reader with a byte-bounded expert cache.
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use kiln_mem::{ExpertCache, ExpertCachePolicy, ExpertCacheStats, ExpertKey};
use kiln_models::ExpertTable;

use crate::{ExpertBlobError, ExpertBlobReader};

/// Reads expert blocks from the blob and retains hot blocks under a strict
/// cache-membership byte budget. Concurrent misses are rechecked after taking
/// the reader lock so only one read of a missing key is needed at a time.
pub struct CachedExpertBlobReader {
    reader: Mutex<ExpertBlobReader>,
    cache: ExpertCache,
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
        })
    }

    fn lock_reader(&self) -> MutexGuard<'_, ExpertBlobReader> {
        self.reader.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn read_expert(
        &self,
        layer: u32,
        expert_id: u32,
    ) -> Result<Vec<u8>, ExpertBlobError> {
        let key = ExpertKey { layer, expert_id };
        if let Some(bytes) = self.cache.get(key) {
            return Ok(bytes);
        }

        let mut reader = self.lock_reader();
        // Another caller may have filled the cache while this caller waited
        // for the single file reader. Avoid duplicate disk reads in that case.
        if let Some(bytes) = self.cache.get_if_present(key) {
            return Ok(bytes);
        }

        let bytes = reader.read_expert(layer, expert_id)?;
        self.cache.insert(key, bytes.clone());
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
        assert!(stats.used_bytes <= stats.capacity_bytes);
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
