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
pub mod scheduler;
pub mod pipeline;
pub mod chat;
pub mod transformer_config;
pub mod transformer_weights;
pub mod transformer;
pub mod dispatch;
pub mod backends;
pub mod kv_cache;

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
pub use scheduler::{
    Scheduler, DefaultScheduler, MonitoredScheduler, StepOutcome,
};

pub use pipeline::{run_once, Loader, PipelineReport, PipelineError};

pub use chat::{Forward, MockForward, Sampler, SamplingStrategy, ChatSession, ChatError};

pub use transformer_config::{TransformerConfig, ConfigError};

pub use transformer_weights::{TransformerWeights, LayerWeights, WeightError};

pub use transformer::{Transformer, TransformerError};
pub use dispatch::{Dispatcher, Operation, Backend, ExecContext, DispatchError, Scratch, TensorId};
pub use kv_cache::{KvCache, LayerKv};
