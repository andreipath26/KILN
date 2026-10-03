//! KILN Core.
//!
//! The heart of the runtime. Contains the Performance Monitor, the Automatic
//! Mode Selector, and the DAG Scheduler. See docs/core-design.md for the
//! full design.
//!
//! This first pass defines the data structures and error types. Behavior
//! lands in the next pass.

pub mod plan;
pub mod revision;
pub mod error;

pub use plan::{ExecutionPlan, Node, Edge, TensorRef, NodeId};
pub use revision::{PlanRevision, RevisionReason};
pub use error::SchedulerError;
