//! Byte-accounted admission bookkeeping for storage tiers.
//!
//! This manager enforces *logical* resident-byte budgets. It does not allocate
//! tensors, perform physical I/O, or measure process RSS/page cache. The actual
//! streaming backend must use these reservations together with OS measurements.

use std::collections::HashMap;

use kiln_hal::ResidencyState;
use crate::error::{EvictError, LoadError};
use crate::policy::{RebalancePolicy, RebalanceReport};
use crate::tier::Tier;

pub type TensorId = u64;

pub trait MemoryManager: Send + Sync {
    fn tier_capacity(&self, tier: Tier) -> u64;
    fn tier_used(&self, tier: Tier) -> u64;
    fn load(&mut self, tensor: TensorId, tier: Tier) -> Result<(), LoadError>;
    fn evict(&mut self, tensor: TensorId) -> Result<(), EvictError>;
    fn residency(&self, tensor: TensorId) -> ResidencyState;
    fn rebalance(&mut self, policy: RebalancePolicy) -> RebalanceReport;
}

/// Admission records are logical accounting only; callers must register a
/// tensor's *real allocated size* (including alignment/padding) before load.
pub struct InMemoryManager {
    capacity: HashMap<Tier, u64>,
    used: HashMap<Tier, u64>,
    residency: HashMap<TensorId, (Tier, u64)>,
    tensor_sizes: HashMap<TensorId, u64>,
}

impl InMemoryManager {
    pub fn new() -> Self {
        let mut capacity = HashMap::new();
        capacity.insert(Tier::L3, 8 * 1024 * 1024);
        // System memory size is a physical ceiling, not a usable RSS budget.
        // The caller must set a lower budget accounting for OS/KV/buffers.
        capacity.insert(Tier::Ram, 16 * 1024 * 1024 * 1024);
        capacity.insert(Tier::Vram, 0);
        capacity.insert(Tier::Ssd, 512 * 1024 * 1024 * 1024);
        Self { capacity, used: HashMap::new(), residency: HashMap::new(), tensor_sizes: HashMap::new() }
    }

    pub fn with_capacity(mut self, tier: Tier, bytes: u64) -> Self {
        self.capacity.insert(tier, bytes);
        self
    }

    /// Register the physical allocation extent before any admission.
    /// Existing resident sizes cannot be changed without eviction.
    pub fn register_tensor_bytes(&mut self, tensor: TensorId, bytes: u64) -> Result<(), LoadError> {
        if bytes == 0 {
            return Err(LoadError::InvalidSize);
        }
        if self.residency.contains_key(&tensor) {
            return Err(LoadError::AlreadyResident);
        }
        self.tensor_sizes.insert(tensor, bytes);
        Ok(())
    }

    pub fn tensor_bytes(&self, tensor: TensorId) -> Option<u64> {
        self.tensor_sizes.get(&tensor).copied()
    }

    pub fn remaining_bytes(&self, tier: Tier) -> u64 {
        self.tier_capacity(tier).saturating_sub(self.tier_used(tier))
    }

    fn account(&mut self, tensor: TensorId, tier: Tier, bytes: u64) {
        let current = self.tier_used(tier);
        self.used.insert(tier, current + bytes); // checked before admission
        self.residency.insert(tensor, (tier, bytes));
    }
}

impl Default for InMemoryManager {
    fn default() -> Self { Self::new() }
}

impl MemoryManager for InMemoryManager {
    fn tier_capacity(&self, tier: Tier) -> u64 {
        *self.capacity.get(&tier).unwrap_or(&0)
    }

    fn tier_used(&self, tier: Tier) -> u64 {
        *self.used.get(&tier).unwrap_or(&0)
    }

    fn load(&mut self, tensor: TensorId, tier: Tier) -> Result<(), LoadError> {
        let bytes = *self.tensor_sizes.get(&tensor).ok_or(LoadError::UnregisteredTensor)?;
        let cap = self.tier_capacity(tier);
        if cap == 0 {
            return Err(LoadError::TierUnavailable(tier));
        }

        if let Some(&(prev, prev_bytes)) = self.residency.get(&tensor) {
            if prev == tier {
                return Ok(());
            }
            if bytes > cap || self.tier_used(tier) > cap - bytes {
                return Err(LoadError::TierFull(tier));
            }
            // All destination capacity checks completed: move atomically.
            let source_used = self.tier_used(prev);
            self.used.insert(prev, source_used - prev_bytes);
            self.account(tensor, tier, bytes);
            return Ok(());
        }

        if bytes > cap || self.tier_used(tier) > cap - bytes {
            return Err(LoadError::TierFull(tier));
        }
        self.account(tensor, tier, bytes);
        Ok(())
    }

    fn evict(&mut self, tensor: TensorId) -> Result<(), EvictError> {
        let (tier, bytes) = self.residency.remove(&tensor).ok_or(EvictError::NotResident)?;
        let current = self.tier_used(tier);
        self.used.insert(tier, current - bytes);
        Ok(())
    }

    fn residency(&self, tensor: TensorId) -> ResidencyState {
        self.residency.get(&tensor)
            .map(|(tier, _)| tier.residency())
            .unwrap_or(ResidencyState::NotLoaded)
    }

    fn rebalance(&mut self, _policy: RebalancePolicy) -> RebalanceReport {
        // No automatic eviction policy implemented: never pretend to migrate.
        RebalanceReport { moved: 0, evicted: 0, duration_nanos: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounts_bytes_and_exact_fit() {
        let mut m = InMemoryManager::new().with_capacity(Tier::Ram, 100);
        m.register_tensor_bytes(1, 60).unwrap();
        m.register_tensor_bytes(2, 40).unwrap();
        m.load(1, Tier::Ram).unwrap();
        m.load(2, Tier::Ram).unwrap();
        assert_eq!(m.tier_used(Tier::Ram), 100);
        assert_eq!(m.remaining_bytes(Tier::Ram), 0);
        m.evict(1).unwrap();
        assert_eq!(m.tier_used(Tier::Ram), 40);
    }

    #[test]
    fn no_overcommit_and_no_mutation_on_failed_move() {
        let mut m = InMemoryManager::new().with_capacity(Tier::Ram, 10).with_capacity(Tier::Ssd, 20);
        m.register_tensor_bytes(1, 11).unwrap();
        m.load(1, Tier::Ssd).unwrap();
        assert_eq!(m.load(1, Tier::Ram), Err(LoadError::TierFull(Tier::Ram)));
        assert_eq!(m.tier_used(Tier::Ssd), 11);
        assert_eq!(m.tier_used(Tier::Ram), 0);
        assert_eq!(m.residency(1), Tier::Ssd.residency());
    }

    #[test]
    fn unregistered_and_zero_and_unavailable_fail_closed() {
        let mut m = InMemoryManager::new();
        assert_eq!(m.load(3, Tier::Ram), Err(LoadError::UnregisteredTensor));
        assert_eq!(m.register_tensor_bytes(3, 0), Err(LoadError::InvalidSize));
        m.register_tensor_bytes(3, 5).unwrap();
        assert_eq!(m.load(3, Tier::Vram), Err(LoadError::TierUnavailable(Tier::Vram)));
    }

    #[test]
    fn repeated_load_is_idempotent_and_resizing_resident_rejected() {
        let mut m = InMemoryManager::new().with_capacity(Tier::Ram, 12);
        m.register_tensor_bytes(2, 7).unwrap();
        m.load(2, Tier::Ram).unwrap();
        m.load(2, Tier::Ram).unwrap();
        assert_eq!(m.tier_used(Tier::Ram), 7);
        assert_eq!(m.register_tensor_bytes(2, 9), Err(LoadError::AlreadyResident));
        m.evict(2).unwrap();
        m.register_tensor_bytes(2, 9).unwrap();
        m.load(2, Tier::Ram).unwrap();
        assert_eq!(m.tier_used(Tier::Ram), 9);
    }

    #[test]
    fn huge_tensor_never_overflows_counter() {
        let mut m = InMemoryManager::new().with_capacity(Tier::Ssd, u64::MAX);
        m.register_tensor_bytes(1, u64::MAX).unwrap();
        m.register_tensor_bytes(2, 1).unwrap();
        m.load(1, Tier::Ssd).unwrap();
        assert_eq!(m.load(2, Tier::Ssd), Err(LoadError::TierFull(Tier::Ssd)));
        assert_eq!(m.tier_used(Tier::Ssd), u64::MAX);
    }
}
