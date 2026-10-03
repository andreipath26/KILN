//! The MemoryManager trait and the in-memory implementation.

use kiln_hal::ResidencyState;

use crate::error::{EvictError, LoadError};
use crate::policy::{RebalancePolicy, RebalanceReport};
use crate::tier::Tier;

/// A tensor is identified by an opaque u64. The registry of names to IDs
/// lives in kiln-core.
pub type TensorId = u64;

/// The manager executes load and evict decisions.
pub trait MemoryManager: Send + Sync {
    /// Capacity of a tier in bytes.
    fn tier_capacity(&self, tier: Tier) -> u64;

    /// Bytes currently used in a tier.
    fn tier_used(&self, tier: Tier) -> u64;

    /// Load a tensor into a tier. Fails if the tier is at capacity and no
    /// eviction is possible.
    fn load(&mut self, tensor: TensorId, tier: Tier) -> Result<(), LoadError>;

    /// Evict a tensor from whatever tier it currently lives in.
    fn evict(&mut self, tensor: TensorId) -> Result<(), EvictError>;

    /// Report the current residency state of a tensor.
    fn residency(&self, tensor: TensorId) -> ResidencyState;

    /// Run a rebalance pass according to the given policy.
    fn rebalance(&mut self, policy: RebalancePolicy) -> RebalanceReport;
}

/// A simple in-memory manager. Tracks residency and usage. Does not perform
/// actual I/O. The I/O layer lives in kiln-io.
pub struct InMemoryManager {
    capacity: std::collections::HashMap<Tier, u64>,
    used: std::collections::HashMap<Tier, u64>,
    residency: std::collections::HashMap<TensorId, Tier>,
}

impl InMemoryManager {
    pub fn new() -> Self {
        let mut capacity = std::collections::HashMap::new();
        capacity.insert(Tier::L3, 8 * 1024 * 1024);
        capacity.insert(Tier::Ram, 16 * 1024 * 1024 * 1024);
        capacity.insert(Tier::Vram, 0);
        capacity.insert(Tier::Ssd, 512 * 1024 * 1024 * 1024);

        Self {
            capacity,
            used: std::collections::HashMap::new(),
            residency: std::collections::HashMap::new(),
        }
    }

    pub fn with_capacity(mut self, tier: Tier, bytes: u64) -> Self {
        self.capacity.insert(tier, bytes);
        self
    }
}

impl Default for InMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager for InMemoryManager {
    fn tier_capacity(&self, tier: Tier) -> u64 {
        *self.capacity.get(&tier).unwrap_or(&0)
    }

    fn tier_used(&self, tier: Tier) -> u64 {
        *self.used.get(&tier).unwrap_or(&0)
    }

    fn load(&mut self, tensor: TensorId, tier: Tier) -> Result<(), LoadError> {
        if let Some(prev) = self.residency.get(&tensor).copied() {
            if prev == tier {
                return Ok(());
            }
            let prev_used = self.used.entry(prev).or_insert(0);
            *prev_used = prev_used.saturating_sub(1);
        }
        let cap = self.tier_capacity(tier);
        let used = self.used.entry(tier).or_insert(0);
        if *used >= cap && cap > 0 {
            return Err(LoadError::TierFull(tier));
        }
        *used += 1;
        self.residency.insert(tensor, tier);
        Ok(())
    }

    fn evict(&mut self, tensor: TensorId) -> Result<(), EvictError> {
        let tier = self.residency.remove(&tensor).ok_or(EvictError::NotResident)?;
        let used = self.used.entry(tier).or_insert(0);
        *used = used.saturating_sub(1);
        Ok(())
    }

    fn residency(&self, tensor: TensorId) -> ResidencyState {
        self.residency
            .get(&tensor)
            .map(|t| t.residency())
            .unwrap_or(ResidencyState::NotLoaded)
    }

    fn rebalance(&mut self, _policy: RebalancePolicy) -> RebalanceReport {
        RebalanceReport {
            moved: 0,
            evicted: 0,
            duration_nanos: 0,
        }
    }
}
