//! Byte-bounded cache for decoded or file-backed MoE expert blocks.
//!
//! The budget limits bytes retained by this cache. Values returned by `get`
//! are owned copies and are not included in the cache budget after they leave it.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

/// Stable identity for an expert within a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Active shared expert leases prevent enough eviction for admission.
    RejectedPinned { entry_bytes: u64, capacity_bytes: u64 },
    /// Adaptive admission retained more frequently used resident experts.
    RejectedCold { entry_bytes: u64, capacity_bytes: u64 },
    Disabled,
}

/// Result of changing the cache byte cap under live expert leases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeOutcome {
    Resized { evicted_entries: usize, capacity_bytes: u64 },
    RejectedPinned { requested_bytes: u64, pinned_bytes: u64 },
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
    /// Number of adaptive ghost-history admission rejections.
    pub cold_rejections: u64,
}

struct Entry {
    bytes: Arc<[u8]>,
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
    cold_rejections: u64,
    /// Bounded history of recent misses, including experts not cached.
    ghost: HashMap<ExpertKey, (u64, u64)>,
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
                cold_rejections: 0,
                ghost: HashMap::new(),
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
            // Age historical misses too: old prompts cannot permanently
            // suppress admission of newly useful experts.
            for (score, _) in inner.ghost.values_mut() {
                *score = (*score / 2).max(1);
            }
        }
        inner.clock
    }

    /// Return a copy of the cached block and update hit/miss/recency statistics.
    pub fn get(&self, key: ExpertKey) -> Option<Vec<u8>> {
        self.get_shared(key).map(|bytes| bytes.as_ref().to_vec())
    }

    /// Share retained bytes without copying the expert allocation.
    /// The returned Arc acts as a lease: pinned entries cannot be evicted
    /// until external handles are released.
    pub fn get_shared(&self, key: ExpertKey) -> Option<Arc<[u8]>> {
        self.get_impl(key, true, true)
    }

    /// Like `get`, but do not count an absent key as a miss. Useful for a
    /// second check after waiting for another thread to finish a disk read.
    pub fn get_if_present(&self, key: ExpertKey) -> Option<Vec<u8>> {
        self.get_shared_if_present(key).map(|bytes| bytes.as_ref().to_vec())
    }

    /// Same as get_shared, but a miss is not counted.
    pub fn get_shared_if_present(&self, key: ExpertKey) -> Option<Arc<[u8]>> {
        self.get_impl(key, false, true)
    }

    fn get_impl(&self, key: ExpertKey, count_miss: bool, count_hit: bool) -> Option<Arc<[u8]>> {
        let mut inner = self.lock();
        let now = Self::tick(&mut inner);
        let value = if let Some(entry) = inner.entries.get_mut(&key) {
            entry.last_access = now;
            entry.frequency = entry.frequency.saturating_add(1);
            Some(Arc::clone(&entry.bytes))
        } else {
            None
        };
        if value.is_some() && count_hit {
            inner.hits = inner.hits.saturating_add(1);
        } else if value.is_none() && count_miss {
            inner.misses = inner.misses.saturating_add(1);
        }
        value
    }

    /// Inspect residency without treating the probe as an actual expert
    /// access. Warmup/placement measurement must not inflate cache hit counts.
    /// The Arc remains a read lease until dropped.
    pub fn peek_shared(&self, key: ExpertKey) -> Option<Arc<[u8]>> {
        self.lock().entries.get(&key)
            .map(|entry| Arc::clone(&entry.bytes))
    }

    /// Record a miss only after a caller has completed its coalescing/recheck path.
    /// This lets wrappers count one outcome per requested expert rather than one per probe.
    pub fn record_miss(&self) {
        let mut inner = self.lock();
        inner.misses = inner.misses.saturating_add(1);
    }

    /// Warm a verified expert only into *unused* cache space. Speculation
    /// cannot evict a hot resident allocation, count as a demand access, or
    /// train adaptive ghost heat. This is distinct from authoritative insert.
    pub fn admit_warm(&self, key: ExpertKey, bytes: Arc<[u8]>) -> bool {
        let mut inner = self.lock();
        if inner.entries.contains_key(&key) {
            return true;
        }
        let cost = bytes.len() as u64;
        if cost == 0 || cost > inner.capacity_bytes.saturating_sub(inner.used_bytes) {
            return false;
        }
        // Prefetched-only entries start as cold; no heat or clock tick.
        let stamp = inner.clock;
        inner.used_bytes += cost;
        inner.entries.insert(key, Entry {
            bytes, last_access: stamp, frequency: 1,
        });
        true
    }

    /// Insert a block while enforcing the retained-cache byte budget.
    ///
    /// Oversized entries are rejected without evicting anything. Re-inserting
    /// an existing key replaces its bytes and counts the key only once.
    pub fn insert(&self, key: ExpertKey, bytes: Vec<u8>) -> InsertOutcome {
        self.insert_shared(key, Arc::from(bytes))
    }

    /// Admit a shared allocation without duplicating the expert bytes.
    /// Existing external Arc references keep the entry pinned in the cache.
    pub fn insert_shared(&self, key: ExpertKey, bytes: Arc<[u8]>) -> InsertOutcome {
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

        // External Arc handles are active read leases. Evicting their cache
        // membership would leave their physical memory allocated but uncharged.
        // Preflight ALL victims first so a rejected insertion changes nothing.
        if inner.entries.get(&key).is_some_and(|old| Arc::strong_count(&old.bytes) > 1) {
            return InsertOutcome::RejectedPinned {
                entry_bytes, capacity_bytes: inner.capacity_bytes,
            };
        }
        let adaptive_score = if inner.policy == ExpertCachePolicy::Adaptive {
            // Count actual arrivals, not prefetch predictions or passive
            // residency probes. Bound ghost metadata independently of the
            // cache byte budget to avoid a memory leak for huge model vocabularies.
            const MAX_GHOST_KEYS: usize = 8192;
            if !inner.ghost.contains_key(&key) && inner.ghost.len() >= MAX_GHOST_KEYS {
                if let Some(oldest) = inner.ghost.iter()
                    .min_by_key(|(key, (_, tick))| (*tick, key.layer, key.expert_id))
                    .map(|(key, _)| *key) {
                    inner.ghost.remove(&oldest);
                }
            }
            let now = Self::tick(&mut inner);
            let heat = inner.ghost.entry(key).or_insert((0, now));
            heat.0 = heat.0.saturating_add(1);
            heat.1 = now;
            heat.0
        } else { 0 };
        let replaced_bytes = inner.entries.get(&key)
            .map_or(0, |old| old.bytes.len() as u64);
        let projected = inner.used_bytes - replaced_bytes;
        let needed = projected.saturating_sub(inner.capacity_bytes - entry_bytes);
        let mut candidates: Vec<_> = inner.entries.iter()
            .filter(|(candidate, entry)| **candidate != key && Arc::strong_count(&entry.bytes) == 1)
            .map(|(candidate, entry)| (*candidate, entry.frequency, entry.last_access, entry.bytes.len() as u64))
            .collect();
        match inner.policy {
            ExpertCachePolicy::Lru => candidates.sort_by_key(|(_, _, last, _)| *last),
            ExpertCachePolicy::Lfu | ExpertCachePolicy::Adaptive =>
                candidates.sort_by_key(|(_, frequency, last, _)| (*frequency, *last)),
        }
        let mut victims = Vec::new();
        let mut reclaimed = 0u64;
        for (candidate, _, _, bytes) in candidates {
            if reclaimed >= needed { break; }
            victims.push(candidate);
            reclaimed = reclaimed.saturating_add(bytes);
        }
        if reclaimed < needed {
            return InsertOutcome::RejectedPinned {
                entry_bytes, capacity_bytes: inner.capacity_bytes,
            };
        }
        if inner.policy == ExpertCachePolicy::Adaptive && needed > 0 {
            // A cold scan must not eject repeatedly selected experts.
            // Require candidate admission heat to reach the hottest victim's
            // recent frequency before changing cache membership.
            let protected_heat = victims.iter()
                .filter_map(|victim| inner.entries.get(victim))
                .map(|entry| entry.frequency)
                .max().unwrap_or(0);
            if adaptive_score < protected_heat {
                inner.cold_rejections = inner.cold_rejections.saturating_add(1);
                return InsertOutcome::RejectedCold {
                    entry_bytes, capacity_bytes: inner.capacity_bytes,
                };
            }
        }
        let replaced = if let Some(old) = inner.entries.remove(&key) {
            inner.used_bytes -= old.bytes.len() as u64;
            true
        } else { false };
        let mut evicted_entries = 0;
        for victim in victims {
            if let Some(old) = inner.entries.remove(&victim) {
                inner.used_bytes -= old.bytes.len() as u64;
                inner.evictions = inner.evictions.saturating_add(1);
                evicted_entries += 1;
            }
        }

        let now = Self::tick(&mut inner);
        inner.used_bytes = inner.used_bytes.saturating_add(entry_bytes);
        let initial_frequency = if inner.policy == ExpertCachePolicy::Adaptive {
            adaptive_score.max(1)
        } else { 1 };
        inner.entries.insert(key, Entry {
            bytes, last_access: now, frequency: initial_frequency,
        });
        if replaced {
            InsertOutcome::Replaced { evicted_entries }
        } else {
            InsertOutcome::Inserted { evicted_entries }
        }
    }

    /// Invalidate an unleased entry. Active shared handles keep residency
    /// charged until released; false also means that a key is pinned.
    pub fn invalidate(&self, key: ExpertKey) -> bool {
        let mut inner = self.lock();
        if inner.entries.get(&key)
            .is_some_and(|entry| Arc::strong_count(&entry.bytes) > 1) {
            return false;
        }
        if let Some(entry) = inner.entries.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(entry.bytes.len() as u64);
            true
        } else {
            false
        }
    }

    /// Clear unleased entries, retaining any still-in-use expert allocations.
    /// Return the number actually removed, not the number initially present.
    pub fn clear(&self) -> usize {
        let mut inner = self.lock();
        let removable: Vec<_> = inner.entries.iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.bytes) == 1)
            .map(|(key, _)| *key)
            .collect();
        for key in &removable {
            if let Some(entry) = inner.entries.remove(key) {
                inner.used_bytes -= entry.bytes.len() as u64;
            }
        }
        removable.len()
    }

    /// Adjust cache budget as memory availability changes. The operation
    /// never frees a block referenced by active CPU/GPU consumers. If the
    /// target cannot be met, leave both capacity and contents unchanged.
    pub fn resize(&self, new_capacity: u64) -> ResizeOutcome {
        let mut inner = self.lock();
        if new_capacity >= inner.used_bytes {
            inner.capacity_bytes = new_capacity;
            return ResizeOutcome::Resized {
                evicted_entries: 0, capacity_bytes: new_capacity
            };
        }
        let mut candidates: Vec<_> = inner.entries.iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.bytes) == 1)
            .map(|(key, entry)| (*key, entry.frequency, entry.last_access,
                                entry.bytes.len() as u64))
            .collect();
        match inner.policy {
            ExpertCachePolicy::Lru => candidates.sort_by_key(|(_, _, last, _)| *last),
            ExpertCachePolicy::Lfu | ExpertCachePolicy::Adaptive =>
                candidates.sort_by_key(|(_, frequency, last, _)| (*frequency, *last)),
        }
        let mut victims = Vec::new();
        let mut reclaim = 0u64;
        for (key, _, _, bytes) in candidates {
            if inner.used_bytes - reclaim <= new_capacity { break; }
            victims.push(key);
            reclaim = reclaim.saturating_add(bytes);
        }
        if inner.used_bytes - reclaim > new_capacity {
            return ResizeOutcome::RejectedPinned {
                requested_bytes: new_capacity,
                pinned_bytes: inner.entries.values()
                    .filter(|entry| Arc::strong_count(&entry.bytes) > 1)
                    .map(|entry| entry.bytes.len() as u64)
                    .sum(),
            };
        }
        for key in &victims {
            if let Some(entry) = inner.entries.remove(key) {
                inner.used_bytes -= entry.bytes.len() as u64;
                inner.evictions = inner.evictions.saturating_add(1);
            }
        }
        inner.capacity_bytes = new_capacity;
        ResizeOutcome::Resized {
            evicted_entries: victims.len(), capacity_bytes: new_capacity
        }
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
            cold_rejections: inner.cold_rejections,
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
    fn lfu_prefers_retaining_a_frequently_accessed_expert() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Lfu);
        cache.insert(key(1), vec![1; 4]);
        cache.insert(key(2), vec![2; 4]);
        assert!(cache.get(key(1)).is_some());
        assert!(cache.get(key(1)).is_some());
        assert_eq!(cache.insert(key(3), vec![3; 4]), InsertOutcome::Inserted { evicted_entries: 1 });
        assert!(cache.get(key(1)).is_some());
        assert!(cache.get(key(2)).is_none());
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
    fn passive_residency_probe_does_not_fake_cache_hits() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Adaptive);
        cache.insert(key(1), vec![4; 8]);
        let first = cache.peek_shared(key(1)).unwrap();
        let second = cache.peek_shared(key(1)).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(cache.peek_shared(key(2)).is_none());
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        drop(first);
        drop(second);
        assert!(cache.get_shared(key(1)).is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn warm_admission_never_evicted_a_hot_entry_or_recorded_demand_heat() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Adaptive);
        cache.insert(key(1), vec![1; 8]);
        assert!(cache.get_shared(key(1)).is_some());
        let before = cache.stats();
        assert!(!cache.admit_warm(key(2), Arc::from(vec![2u8;8])));
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.stats().hits, before.hits);
        assert_eq!(cache.stats().misses, before.misses);
        assert_eq!(cache.stats().cold_rejections, before.cold_rejections);
        assert_eq!(cache.get(key(1)), Some(vec![1;8]));
        assert!(!cache.admit_warm(key(3), Arc::from(Vec::<u8>::new())));
        assert!(cache.invalidate(key(1)));
        assert!(cache.admit_warm(key(2), Arc::from(vec![2u8;8])));
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn shared_cache_hit_reuses_allocation_without_copy() {
        let cache = ExpertCache::new(16, ExpertCachePolicy::Lru);
        let value: Arc<[u8]> = Arc::from(vec![1, 2, 3, 4]);
        assert_eq!(cache.insert_shared(key(1), Arc::clone(&value)), InsertOutcome::Inserted { evicted_entries: 0 });
        let first = cache.get_shared(key(1)).unwrap();
        let second = cache.get_shared(key(1)).unwrap();
        assert!(Arc::ptr_eq(&value, &first));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.get(key(1)), Some(vec![1, 2, 3, 4]));
        assert_eq!(cache.stats().used_bytes, 4);
    }

    #[test]
    fn evicting_shared_entry_preserves_external_handle() {
        let cache = ExpertCache::new(4, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![9; 4]);
        let held = cache.get_shared(key(1)).unwrap();
        assert_eq!(cache.insert(key(2), vec![8; 4]), InsertOutcome::RejectedPinned {
            entry_bytes: 4, capacity_bytes: 4,
        });
        assert_eq!(held.as_ref(), &[9, 9, 9, 9]);
        assert_eq!(cache.stats().used_bytes, 4);
        assert!(cache.get_shared(key(1)).is_some());
        // The external Arc acts as a lease: residency stays charged until release.
        drop(held);
        assert_eq!(cache.insert(key(2), vec![8; 4]), InsertOutcome::Inserted { evicted_entries: 1 });
        assert!(cache.get_shared(key(1)).is_none());
    }

    #[test]
    fn live_expert_lease_prevents_physical_overcommit_and_partial_eviction() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 4]);
        cache.insert(key(2), vec![2; 4]);
        let pin = cache.get_shared(key(1)).unwrap();
        let result = cache.insert(key(3), vec![3; 8]);
        assert_eq!(result, InsertOutcome::RejectedPinned {
            entry_bytes: 8, capacity_bytes: 8,
        });
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
        drop(pin);
        assert_eq!(cache.insert(key(3), vec![3; 8]),
                   InsertOutcome::Inserted { evicted_entries: 2 });
        assert_eq!(cache.stats().used_bytes, 8);
    }

    #[test]
    fn replacing_a_leased_expert_is_rejected_without_invalidating_reader() {
        let cache = ExpertCache::new(16, ExpertCachePolicy::Adaptive);
        cache.insert(key(1), vec![5; 8]);
        let pin = cache.get_shared(key(1)).unwrap();
        assert_eq!(cache.insert(key(1), vec![6; 8]), InsertOutcome::RejectedPinned {
            entry_bytes: 8, capacity_bytes: 16
        });
        assert_eq!(pin.as_ref(), &[5; 8]);
        assert_eq!(cache.stats().used_bytes, 8);
        drop(pin);
        assert_eq!(cache.insert(key(1), vec![6; 8]),
                   InsertOutcome::Replaced { evicted_entries: 0 });
    }

    #[test]
    fn adaptive_admission_resists_one_off_scan_then_promotes_repeated_expert() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Adaptive);
        assert_eq!(cache.insert(key(1), vec![1; 4]),
                   InsertOutcome::Inserted { evicted_entries: 0 });
        assert_eq!(cache.insert(key(2), vec![2; 4]),
                   InsertOutcome::Inserted { evicted_entries: 0 });
        for _ in 0..8 {
            assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
            assert_eq!(cache.get(key(2)), Some(vec![2; 4]));
        }
        assert_eq!(cache.insert(key(3), vec![3; 4]),
                   InsertOutcome::RejectedCold { entry_bytes: 4, capacity_bytes: 8 });
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.get(key(1)), Some(vec![1; 4]));
        assert_eq!(cache.get(key(2)), Some(vec![2; 4]));
        let mut accepted = false;
        for _ in 0..24 {
            if matches!(cache.insert(key(3), vec![3; 4]),
                        InsertOutcome::Inserted { .. }) {
                accepted = true;
                break;
            }
        }
        assert!(accepted, "repeated expert must eventually displace one old resident");
        assert_eq!(cache.get(key(3)), Some(vec![3; 4]));
        assert!(cache.stats().cold_rejections > 0);
        assert!(cache.stats().used_bytes <= 8);
    }

    #[test]
    fn adaptive_history_is_bounded_under_unique_expert_scan() {
        let cache = ExpertCache::new(8, ExpertCachePolicy::Adaptive);
        for id in 0..10_000 {
            cache.insert(key(id), vec![id as u8; 4]);
        }
        assert!(cache.lock().ghost.len() <= 8192);
        assert!(cache.stats().used_bytes <= cache.stats().capacity_bytes);
    }

    #[test]
    fn resizing_cache_is_atomic_and_respects_pinned_expert_leases() {
        let cache = ExpertCache::new(16, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 8]);
        cache.insert(key(2), vec![2; 8]);
        let held = cache.get_shared(key(1)).unwrap();
        assert_eq!(cache.resize(4), ResizeOutcome::RejectedPinned {
            requested_bytes: 4, pinned_bytes: 8
        });
        assert_eq!(cache.stats().capacity_bytes, 16);
        assert_eq!(cache.stats().used_bytes, 16);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(held.as_ref(), &[1; 8]);
        drop(held);
        assert_eq!(cache.resize(4), ResizeOutcome::Resized {
            evicted_entries: 2, capacity_bytes: 4
        });
        assert_eq!(cache.stats().used_bytes, 0);
        assert_eq!(cache.stats().capacity_bytes, 4);
        assert_eq!(cache.resize(32), ResizeOutcome::Resized {
            evicted_entries: 0, capacity_bytes: 32
        });
        assert_eq!(cache.insert(key(3), vec![3; 16]),
                   InsertOutcome::Inserted { evicted_entries: 0 });
    }

    #[test]
    fn lru_does_not_use_adaptive_hysteresis() {
        let cache = ExpertCache::new(4, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 4]);
        for _ in 0..20 { let _ = cache.get(key(1)); }
        assert_eq!(cache.insert(key(2), vec![2; 4]),
                   InsertOutcome::Inserted { evicted_entries: 1 });
        assert_eq!(cache.stats().cold_rejections, 0);
    }

    #[test]
    fn invalidate_and_clear_respect_live_leases() {
        let cache = ExpertCache::new(16, ExpertCachePolicy::Lru);
        cache.insert(key(1), vec![1; 8]);
        cache.insert(key(2), vec![2; 8]);
        let lease = cache.get_shared(key(1)).unwrap();
        assert!(!cache.invalidate(key(1)));
        assert_eq!(cache.clear(), 1);
        assert_eq!(cache.stats().used_bytes, 8);
        assert_eq!(cache.get(key(1)), Some(vec![1; 8]));
        drop(lease);
        assert_eq!(cache.clear(), 1);
        assert_eq!(cache.stats().used_bytes, 0);
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
