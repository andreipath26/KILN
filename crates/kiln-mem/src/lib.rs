//! KILN Memory Hierarchy Manager.
//!
//! The manager owns the four storage tiers and executes load and evict
//! decisions made by the scheduler. It does not decide what to load.
//!
//! See docs/architecture.md, Layer 2, for the design rationale.

pub mod tier;
pub mod manager;
pub mod policy;
pub mod error;

pub use tier::Tier;
pub use manager::{MemoryManager, TensorId};
pub use policy::{RebalancePolicy, RebalanceReport};
pub use error::{LoadError, EvictError};
