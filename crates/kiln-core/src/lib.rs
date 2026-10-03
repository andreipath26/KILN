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
pub mod monitor;
pub mod selector;

pub use plan::{ExecutionPlan, Node, Edge, TensorRef, NodeId};
pub use revision::{PlanRevision, RevisionReason};
pub use error::SchedulerError;
pub use monitor::{
    PerformanceMonitor, PerformanceEnvelope, ThermalHistory, ThrottlePrediction,
    Sample, LinuxMonitor, MockMonitor,
};
pub use selector::{
    ModeSelector, DefaultSelector, SelectionOutcome, LadderRung, AlgorithmReason,
};
