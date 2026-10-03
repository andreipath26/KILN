//! Rebalance policies and the rebalance report.

use serde::{Deserialize, Serialize};

/// What the rebalance pass optimizes for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancePolicy {
    /// Move tensors to the tier that minimizes total access latency.
    MinimizeLatency,
    /// Move tensors to the tier that minimizes SSD writes.
    MinimizeWrite,
    /// Move tensors to the tier that minimizes total power draw.
    MinimizePower,
    /// Custom policy supplied by the scheduler.
    Custom,
}

/// Result of a single rebalance pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebalanceReport {
    /// Number of tensors moved between tiers.
    pub moved: u32,
    /// Number of tensors evicted entirely.
    pub evicted: u32,
    /// Wall-clock duration of the rebalance pass.
    pub duration_nanos: u64,
}
