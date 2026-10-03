//! Thermal profile.
//!
//! Two plans ship with every model: the cold-start plan and the
//! steady-state plan. The runtime loads the cold-start plan, monitors
//! thermal state, and transitions to the steady-state plan when throttling
//! is detected.

use serde::{Deserialize, Serialize};

/// Per-tier memory budget for one plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierBudget {
    /// Hardware tier identifier. 0 through 6.
    pub tier: u32,

    /// Maximum RAM in bytes this plan may use.
    pub max_ram_bytes: u64,

    /// Maximum VRAM in bytes, if a GPU is present.
    pub max_vram_bytes: u64,

    /// Expected throughput in tokens per second at steady state.
    pub expected_tokens_per_sec: f32,
}

/// A single execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Human-readable name. Example: "cold-start" or "steady-state".
    pub name: String,

    /// Per-tier budgets.
    pub budgets: Vec<TierBudget>,

    /// Algorithm preference order. First is tried first.
    pub algorithm_preference: Vec<String>,
}

/// The thermal profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalProfile {
    /// The plan used when the machine is at baseline temperature.
    pub cold_start: Plan,

    /// The plan used when throttling is detected.
    pub steady_state: Plan,
}
