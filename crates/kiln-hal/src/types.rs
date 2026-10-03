//! Core types shared across every KILN crate.
//!
//! These enums and identifiers are the vocabulary of the runtime. They do not
//! carry behavior. They carry meaning.

use serde::{Deserialize, Serialize};

/// Unique identifier for a registered backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub u32);

/// The kind of compute unit a backend represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    WebGpu,
    OpenVino,
    CoreMl,
    Qnn,
    Wasm,
}

/// The numeric format a tensor is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Int4,
    Tq1_0,
    Tq2_0,
    Tq2p,
    Tq3p,
}

/// Where a tensor currently lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyState {
    InL3,
    InRam,
    InVram,
    OnSsd,
    NotLoaded,
}

/// The current thermal condition of a backend or the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThermalState {
    Cold,
    Warm,
    Throttling,
    SeverelyThrottling,
}

/// The type of computation a node performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    MatMul,
    Attention,
    ExpertRoute,
    ExpertCompute,
    Norm,
    Activation,
    KvRead,
    KvWrite,
    Embed,
    Sample,
}

/// The algorithm selected for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Algorithm {
    Diffusion,
    TernaryAr,
    MoeStream,
    SpecDecode,
    Dense,
}
