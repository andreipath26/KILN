//! KVDRIVE layout index for the KV cache.
//!
//! Two-level packing:
//!   Level 1: semantic-contiguity packing. KV entries frequently attended
//!            together are placed in sequential SSD extents.
//!   Level 2: layer-head partitioning. Extents are partitioned by layer and
//!            attention head, preserving structural locality.

use serde::{Deserialize, Serialize};

/// A single KV extent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvExtent {
    pub layer: u32,
    pub head: u32,
    /// Block IDs placed in this extent, in order.
    pub block_ids: Vec<u32>,
    /// Byte offset in the KV layout file.
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// The full KVDRIVE layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvLayout {
    pub layout_version: u32,
    pub block_size_bytes: u32,
    pub total_extents: u32,
    pub extents: Vec<KvExtent>,
}
