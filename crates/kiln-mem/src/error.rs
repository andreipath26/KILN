//! Error types for the Memory Manager.

use crate::tier::Tier;

/// Errors from load operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The target tier has no capacity left and no eviction is possible.
    TierFull(Tier),
    /// The tier requested does not exist on this machine.
    TierUnavailable(Tier),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::TierFull(t) => write!(f, "tier full: {:?}", t),
            LoadError::TierUnavailable(t) => write!(f, "tier unavailable: {:?}", t),
        }
    }
}

impl std::error::Error for LoadError {}

/// Errors from evict operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictError {
    /// The tensor is not resident anywhere.
    NotResident,
}

impl std::fmt::Display for EvictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvictError::NotResident => write!(f, "tensor not resident"),
        }
    }
}

impl std::error::Error for EvictError {}
