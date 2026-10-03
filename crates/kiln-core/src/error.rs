//! Errors from the Scheduler and Selector.

use kiln_hal::BackendId;
use kiln_mem::{EvictError, LoadError};

/// Errors that can occur during plan execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// A node in the plan references a backend that is not registered.
    UnknownBackend(BackendId),
    /// A node's inputs could not be loaded into the required tier.
    LoadFailed(LoadError),
    /// A tensor could not be evicted when needed.
    EvictFailed(EvictError),
    /// The plan contains a cycle. This should never happen.
    PlanCycle,
    /// The plan is empty.
    EmptyPlan,
    /// A backend failed to execute a node.
    BackendFailure(BackendId),
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::UnknownBackend(id) => write!(f, "unknown backend: {:?}", id),
            SchedulerError::LoadFailed(e) => write!(f, "load failed: {}", e),
            SchedulerError::EvictFailed(e) => write!(f, "evict failed: {}", e),
            SchedulerError::PlanCycle => write!(f, "plan contains a cycle"),
            SchedulerError::EmptyPlan => write!(f, "plan is empty"),
            SchedulerError::BackendFailure(id) => write!(f, "backend failed: {:?}", id),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<LoadError> for SchedulerError {
    fn from(e: LoadError) -> Self {
        SchedulerError::LoadFailed(e)
    }
}

impl From<EvictError> for SchedulerError {
    fn from(e: EvictError) -> Self {
        SchedulerError::EvictFailed(e)
    }
}
