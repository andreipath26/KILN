//! Per-tensor quantization specification.

use serde::{Deserialize, Serialize};

use kiln_hal::Precision;

/// A quantization specification for one or more tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSpec {
    /// A glob pattern matching tensor names. Example: "blk.*.ffn_gate.*".
    pub tensor_pattern: String,

    /// The precision used for these tensors.
    pub precision: Precision,

    /// Optional group size for grouped quantization.
    pub group_size: Option<u32>,

    /// Optional calibration metadata. Free-form for now.
    pub calibration: Option<String>,
}
