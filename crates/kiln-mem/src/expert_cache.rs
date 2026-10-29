//! Byte-bounded cache for decoded or file-backed MoE expert blocks.
//!
//! The budget limits bytes retained by this cache. Values returned by `get`
//! are owned copies and are not included in the cache budget after they leave it.
use std::collections::HashMap;
use std::sync::Mutex;

/// Stable identity for an expert within a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert_id: u32,
}

/// Eviction policy used when the byte budget is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCachePolicy {
    /// Evict the least-recently-used entry.
    Lru,
    /// Evict the least-frequently-used entry, breaking ties by oldest access.
    Lfu,
    /// LFU with periodic frequency aging and recency tie-breaking.
    Adaptive,
}

/// Result of inserting an expert block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    Inserted { evicted_entries: usize },
    Replaced { evicted_entries: usize },
    RejectedOversized { entry_bytes: u64, capacity_bytes: u64 },
    Disabled,
}

/// Snapshot of cache counters and byte accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct Entry {
    bytes: Vec<u8>,
    last_access: u64,
    frequency: u64,
}

struct Inner {
    capacity_bytes: u64,
    used_bytes: u64,
    clock: u64,
    policy: ExpertCachePolicy,
    entries: HashMap<ExpertKey, Entry>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Thread-safe, byte-bounded cache. Operations are serialized under one mutex;
/// disk reads should happen outside the cache lock and then be inserted here.
pub struct ExpertCache {
    inner: Mutex<Inner>,
}

impl ExpertCache {
    pub fn new(capacity_bytes: u64, policy: ExpertCachePolicy) -> Self {
        Self {
            inner: Mutex::new(Inner {
                capacity_bytes,
                used_bytes: 0,
                clock: 0,
                policy,
                entries: HashMap::new(),
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn tick(inner: &mut Inner) -> u64 {
        inner.clock = inner.clock.saturating_add(1);
        if inner.policy == ExpertCachePolicy::Adaptive && inner.clock % 4096 == 0 {
            for entry in inner.entries.values_mut() {
                entry.frequency = (entry.frequency / 2).max(1);
            }
        }
        inner.clock
    }

    /// Return a copy of the cached block and update hit/miss/recency statistics.
    pub fn get(&self, key: ExpertKey) -> Option<Vec<u8>> {
        self.get_impl(key, true)
    }

    /// Like `get`, but do not count an absent key as a miss. Useful for a
    /// second check after waiting for another thread to finish a disk read.
    pub fn get_if_present(&self, key: ExpertKey) -> Option<Vec<u8>> {
        self.get_impl(key, false)
    }

    fn get_impl(&self, key: ExpertKey, count_miss: bool) -> Option<Vec<u8>> {
        let mut inner = self.lock();
        let now = Self::tick(&mut inner);
        let value = if let Some(entry) = inner.entries.get_mut(&key) {
            entry.last_access = now;
            entry.frequency = entry.frequency.saturating_add(1);
            Some(entry.bytes.clone())
        } else {
            None
        };
        if value.is_some() {
            inner.hits = inner.hits.saturating_add(1);
        } else if count_miss {
            inner.misses = inner.misses.saturating_add(1);
        }
        value
    }

    /// Insert a block while enforcing the retained-cache byte budget.
    ///
    /// Oversized entries are rejected without evicting anything. Re-inserting
    /// an existing key replaces its bytes and counts the key only once.
    pub fn insert(&self, key: ExpertKey, bytes: Vec<u8>) -> InsertOutcome {
        let entry_bytes = bytes.len() as u64;
        let mut inner = self.lock();
        if inner.capacity_bytes == 0 {
            return InsertOutcome::Disabled;
        }
        if entry_bytes > inner.capacity_bytes {
            return InsertOutcome::RejectedOversized {
                entry_bytes,
                capacity_bytes: inner.capacity_bytes,
            };
        }

        let replaced = if let Some(old) = inner.entries.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(old.bytes.len() as u64);
            true
        } else {
            false
        };

        let mut evicted_entries = 0;
        let target_used = inner.capacity_bytes - entry_bytes;
        while inner.used_bytes > target_used {
            let victim = match inner.policy {
                ExpertCachePolicy::Lru => inner.entries.iter()
                    .min_by_key(|(_, entry)| entry.last_access)
                    .map(|(key, _)| *key),
                ExpertCachePolicy::Lfu | ExpertCachePolicy::Adaptive => inner.entries.iter()
                    .min_by_key(|(_, entry)| (entry.frequency, entry.last_access))
                    .map(|(key, _)| *key),
            };
            let Some(victim) = victim else { break };
            if let Some(old) = inner.entries.remove(&victim) {
                inner.used_bytes = inner.used_bytes.saturating_sub(old.bytes.len() as u64);
                inner.evictions = inner.evictions.saturating_add(1);
                evicted_entries += 1;
            }
        }

        let now = Self::tick(&mut inner);
        inner.used_bytes = inner.used_bytes.saturating_add(entry_bytes);
        inner.entries.insert(key, Entry { bytes, last_access: now, frequency: 1 });
        if replaced {
            InsertOutcome::Replaced { evicted_entries }
        } else {
            InsertOutcome::Inserted { evicted_entries }
        }
    }

    /// Remove a single cached expert. Returns whether it was present.
    pub fn invalidate(&self, key: ExpertKey) -> bool {
        let mut inner = self.lock();
        if let Some(entry) = inner.entries.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(entry.bytes.len() as u64);
            true
        } else {
            false
        }
    }

    /// Remove every entry while retaining cumulative hit/miss/eviction counters.
    pub fn clear(&self) -> usize {
        let mut inner = self.lock();
        let removed = inner.entries.len();
        inner.entries.clear();
        inner.used_bytes = 0;
        removed
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let inner = self.lock();
        ExpertCacheStats {
            capacity_bytes: inner.capacity_bytes,
            used_bytes: inner.used_bytes,
            entries: inner.entries.len(),
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn key(id: u32) -> ExpertKey {
        ExpertKey { layer: 0, expert_id: id }
    }

    #[test]
    fn lru_evicts_least_recently_used_entry_to_stay_under_byte_cap() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 4]);
        cache.insert(key(2), vec![2; 4]);
        assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
        let outcome = cache.insert(key(3), vec![3; 4]);
        assert_eq!(outcome, InsertOutcome::Inserted { evicted_entries: 1 });
        assert!(cache.get(key(2)).is_none());
        assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.stats().entries, 2);
    }

    #[test]
    fn oversized_entry_is_rejected_without_evicting_existing_entries() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 4]);
        assert_eq!(cache.insert(key(2), vec![2; 9]), InsertOutcome::RejectedOversized {
            entry_bytes: 9, capacity_bytes: 8
        });
        assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
        assert_eq!(cache.stats().used_bytes, 4);
    }

    #[test]
    fn replacing_key_does_not_double_count_bytes_or_entries() {
        let cache = ExpertCache::new(16, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 4]);
        assert_eq!(cache.insert(key(1), vec![2; 6]), InsertOutcome::Replaced { evicted_entries: 0 });
        assert_eq!(cache.get(key(1)), Some(vec![2; 6]));
        assert_eq!(cache.stats().used_bytes, 6);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn invalidation_and_clear_release_retained_bytes() {
        let cache = ExpertCache::new(32, ExpertCachePolicy::Lfu);
        cache.insert(key(1), vec![1; 5]);
        cache.insert(key(2), vec![2; 7]);
        assert!(cache.invalidate(key(1)));
        assert!(!cache.invalidate(key(1)));
        assert_eq!(cache.stats().used_bytes, 7);
        assert_eq!(cache.clear(), 1);
        assert_eq!(cache.stats().used_bytes, 0);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn zero_capacity_disables_retention() {
        let cache = ExpertCache::new(0, ExpertCachePolicy::Lru);
        assert_eq!(cache.insert(key(1), vec![1]), InsertOutcome::Disabled);
        assert!(cache.get(key(1)).is_none());
        assert_eq!(cache.stats().used_bytes, 0);
    }

    #[test]
    fn concurrent_duplicate_inserts_keep_one_entry_and_correct_accounting() {
        let cache = Arc::new(ExpertCache::new(4096, ExpertCachePolicy::Lru));
        let mut workers = Vec::new();
        for value in 0..8u8 {
            let cache = Arc::clone(&cache);
            workers.push(thread::spawn(move || {
                for _ in 0..100 {
                    cache.insert(key(1), vec![value; 128]);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.used_bytes, 128);
        assert!(cache.get(key(1)).is_some());
    }

    #[test]
    fn concurrent_distinct_inserts_never_exceed_byte_budget() {
        let cache = Arc::new(ExpertCache::new(4096, ExpertCachePolicy::Adaptive));
        let mut workers = Vec::new();
        for worker_id in 0..8u32 {
            let cache = Arc::clone(&cache);
            workers.push(thread::spawn(move || {
                for item in 0..100u32 {
                    cache.insert(key(worker_id * 100 + item), vec![worker_id as u8; 64]);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let stats = cache.stats();
        assert!(stats.used_bytes <= stats.capacity_bytes);
        assert_eq!(stats.entries as u64 * 64, stats.used_bytes);
    }
}
