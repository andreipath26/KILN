//! Plan revisions. Every time the selector changes the plan, a revision is
//! recorded. The change log is append-only.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Why the plan was revised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevisionReason {
    /// Initial plan at model load.
    LoadTime,
    /// Thermal state dropped below threshold.
    ThermalThrottle,
    /// Thermal state recovered above threshold for 60 seconds.
    ThermalRecovery,
    /// User changed a setting explicitly.
    ManualOverride,
    /// A backend failed and the plan was rebuilt without it.
    ErrorFallback,
}

/// One entry in the change log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRevision {
    pub revision: u64,
    pub reason: RevisionReason,
    #[serde(with = "serde_system_time")]
    pub timestamp: SystemTime,
    pub previous_throughput_fraction: f32,
    pub current_throughput_fraction: f32,
}

mod serde_system_time {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let d = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        d.as_secs().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}
