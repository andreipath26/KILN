//! Top-level manifest for a UMF container.
//!
//! This is the in-memory representation of manifest.json. Every field maps
//! directly to a JSON key. Load order matters: the manifest is read first,
//! then the referenced files are loaded according to the layout indices.

use serde::{Deserialize, Serialize};

use crate::quantization::QuantizationSpec;
use crate::thermal::ThermalProfile;

/// The architecture family of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    Dense,
    Moe,
    Ssm,
    Hybrid,
}

/// The full manifest for a UMF container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Schema version. Bumped on breaking changes.
    pub schema_version: u32,

    /// Model name as it appears in the catalog.
    pub name: String,

    /// Source model URL. Informational.
    pub source_url: String,

    /// Checksum of the original source model. Informational.
    pub source_checksum: String,

    /// License identifier. SPDX where possible.
    pub license: String,

    /// Whether redistribution is permitted by the license.
    pub redistribution_allowed: bool,

    /// Architecture family.
    pub architecture: Architecture,

    /// Total parameter count. Informational.
    pub total_parameters: u64,

    /// Active parameter count per token for MoE. Zero for dense.
    pub active_parameters_per_token: u64,

    /// Number of expert layers. Zero for dense.
    pub expert_layers: u32,

    /// Number of experts per layer. Zero for dense.
    pub experts_per_layer: u32,

    /// Quantization specification for each tensor group.
    pub quantization: Vec<QuantizationSpec>,

    /// Thermal profile with cold-start and steady-state plans.
    pub thermal_profile: ThermalProfile,

    /// Path to the expert layout index, relative to the UMF root.
    pub experts_index_path: Option<String>,

    /// Path to the expert-contiguous weight blob, relative to UMF root.
    pub experts_blob_path: Option<String>,

    /// Path to the KVDRIVE layout index, relative to UMF root.
    pub kv_layout_path: Option<String>,

    /// Path to the primary weights file (GGUF), relative to UMF root.
    pub weights_path: String,

    /// Path to the ML cache directory, relative to UMF root.
    pub ml_cache_dir: Option<String>,
}
