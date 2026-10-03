//! Reference to the FlashMoE ML cache for a single layer.

use serde::{Deserialize, Serialize};

/// A reference to one layer's ML cache file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlCacheRef {
    /// Layer index.
    pub layer: u32,

    /// Path to the cache file, relative to the UMF root.
    pub path: String,

    /// Size in bytes. Roughly 113 KB for a 30B MoE layer.
    pub size_bytes: u64,

    /// Checksum of the cache file.
    pub checksum: String,

    /// Input feature count used at training time.
    pub input_features: u32,

    /// Number of experts the cache predicts over.
    pub num_experts: u32,
}
