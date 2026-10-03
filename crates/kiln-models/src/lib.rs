//! KILN Unified Model Format.
//!
//! The UMF is a directory layout that wraps a GGUF model with KILN-specific
//! metadata. This crate defines the Rust types that correspond to the
//! manifest.json file and the layout indices.
//!
//! See docs/architecture.md, section "Unified Model Format", for the
//! directory structure and the rationale for each field.

pub mod manifest;
pub mod quantization;
pub mod experts;
pub mod kv_layout;
pub mod thermal;
pub mod ml_cache;
pub mod synthetic;
pub mod gguf;

pub use manifest::{Architecture, ModelManifest};
pub use quantization::QuantizationSpec;
pub use experts::ExpertTable;
pub use kv_layout::KvLayout;
pub use thermal::ThermalProfile;
pub use ml_cache::MlCacheRef;

pub use synthetic::{SyntheticModel, write_synthetic, read_synthetic, SyntheticError};

pub use gguf::{GgufError, GgufHeader, GgufType, GgufTensorInfo, GgufValue, GgufValueType, read_header, parse_metadata, parse_tensor_table};
