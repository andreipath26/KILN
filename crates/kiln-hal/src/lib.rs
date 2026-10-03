//! KILN Hardware Abstraction Layer.
//!
//! This crate defines the interfaces that every hardware backend implements
//! and the Constraint-Aware Registry that the scheduler queries.
//!
//! See docs/architecture.md, Layers 1 and 3, for the design rationale.

pub mod types;
pub mod backend;
pub mod registry;

pub use types::{
    BackendId, DeviceType, Precision, ResidencyState, ThermalState,
};
pub use backend::Backend;
pub use registry::Registry;
