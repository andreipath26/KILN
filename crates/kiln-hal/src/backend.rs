//! The Backend trait.
//!
//! Every compute unit implements this trait and registers itself with the
//! Registry. The trait carries no execution logic. It only declares what a
//! backend supports, what it costs to use, and what state it is in.

use crate::types::{
    BackendId, DeviceType, Op, Precision, ResidencyState, ThermalState,
};

/// Describes a backend's memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryModel {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub bandwidth_bytes_per_sec: u64,
}

/// A request to move a tensor from one precision to another on a given backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionCost {
    pub from: Precision,
    pub to: Precision,
    pub cost_nanos: u64,
}

/// A single step in a conversion path that crosses backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionStep {
    pub backend: BackendId,
    pub from: Precision,
    pub to: Precision,
    pub cost_nanos: u64,
}

/// The full path needed to convert a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPath {
    pub steps: Vec<ConversionStep>,
    pub total_cost_nanos: u64,
}

/// Every backend in the runtime implements this trait.
pub trait Backend: Send + Sync {
    /// Stable identifier for this backend instance.
    fn id(&self) -> BackendId;

    /// The kind of compute unit.
    fn device_type(&self) -> DeviceType;

    /// Operations this backend can execute.
    fn supported_ops(&self) -> &[Op];

    /// Numeric formats this backend can execute.
    fn supported_precisions(&self) -> &[Precision];

    /// Current memory model of this backend.
    fn memory_model(&self) -> MemoryModel;

    /// Bytes currently free.
    fn available_memory(&self) -> u64;

    /// Nominal bytes per second this backend can move.
    fn device_bandwidth(&self) -> u64;

    /// Cost to convert a tensor from one precision to another on this backend.
    fn conversion_cost(&self, from: Precision, to: Precision) -> Option<u64>;

    /// Where a tensor currently lives according to this backend.
    fn residency_state(&self, tensor_id: u64) -> ResidencyState;

    /// Nanoseconds of latency to transfer a tensor to another backend.
    fn transfer_latency(&self, to: BackendId) -> u64;

    /// Nanoseconds of synchronization overhead for a dispatch on this backend.
    fn synchronization_overhead(&self) -> u64;

    /// Current thermal state of this backend.
    fn thermal_state(&self) -> ThermalState;
}
