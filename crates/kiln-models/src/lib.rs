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

pub use manifest::{Architecture, ModelManifest};
pub use quantization::QuantizationSpec;
pub use experts::ExpertTable;
pub use kv_layout::KvLayout;
pub use thermal::ThermalProfile;
pub use ml_cache::MlCacheRef;

pub use synthetic::{SyntheticModel, write_synthetic, read_synthetic, SyntheticError};
