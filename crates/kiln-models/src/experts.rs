//! Expert layout index for MoE models.
//!
//! The experts.bin file is expert-contiguous. This index maps expert IDs to
//! byte ranges. It is a flat list sorted by (layer, expert_id) so that the
//! whole-layer prefill path can load all experts for one layer with a single
//! sequential read pattern.

use serde::{Deserialize, Serialize};

/// A single expert byte range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRange {
    pub layer: u32,
    pub expert_id: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// The full expert table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertTable {
    /// Format version of the blob.
    pub blob_version: u32,

    /// Total bytes in the blob.
    pub blob_size: u64,

    /// All expert ranges, sorted by (layer, expert_id).
    pub ranges: Vec<ExpertRange>,
}
