//! The four storage tiers managed by the runtime.

use serde::{Deserialize, Serialize};

/// A physical storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// CPU L3 cache. Latency roughly 1 nanosecond.
    L3,
    /// System RAM. Latency roughly 100 nanoseconds on DDR4.
    Ram,
    /// NVMe SSD. Latency roughly 10 to 100 microseconds.
    Ssd,
    /// Optional GPU VRAM. Latency roughly 100 to 500 nanoseconds.
    Vram,
}

impl Tier {
    /// Nominal latency in nanoseconds for this tier.
    pub fn latency_nanos(&self) -> u64 {
        match self {
            Tier::L3 => 1,
            Tier::Ram => 100,
            Tier::Vram => 300,
            Tier::Ssd => 50_000,
        }
    }

    /// The residency state that corresponds to being loaded in this tier.
    pub fn residency(&self) -> kiln_hal::ResidencyState {
        match self {
            Tier::L3 => kiln_hal::ResidencyState::InL3,
            Tier::Ram => kiln_hal::ResidencyState::InRam,
            Tier::Vram => kiln_hal::ResidencyState::InVram,
            Tier::Ssd => kiln_hal::ResidencyState::OnSsd,
        }
    }
}
