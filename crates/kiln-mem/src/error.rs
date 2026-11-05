//! Errors from logical memory-residency accounting.
use crate::tier::Tier;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    TierFull(Tier),
    TierUnavailable(Tier),
    UnregisteredTensor,
    InvalidSize,
    AlreadyResident,
}
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::TierFull(t) => write!(f, "tier full: {:?}", t),
            LoadError::TierUnavailable(t) => write!(f, "tier unavailable: {:?}", t),
            LoadError::UnregisteredTensor => write!(f, "tensor size not registered"),
            LoadError::InvalidSize => write!(f, "tensor size must be nonzero"),
            LoadError::AlreadyResident => write!(f, "evict tensor before changing its registered size"),
        }
    }
}
impl std::error::Error for LoadError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictError { NotResident }
impl std::fmt::Display for EvictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { EvictError::NotResident => write!(f, "tensor not resident") }
    }
}
impl std::error::Error for EvictError {}
