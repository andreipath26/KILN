//! I/O request and result types.

use kiln_mem::{TensorId, Tier};

/// Direction of an I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOp {
    /// Read from storage into memory.
    Read,
    /// Write from memory to storage.
    Write,
    /// Synchronize a previously written buffer.
    Sync,
}

/// A single I/O request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoRequest {
    /// What kind of operation this is.
    pub op: IoOp,
    /// The tensor being moved.
    pub tensor: TensorId,
    /// Source tier, if reading.
    pub from: Option<Tier>,
    /// Destination tier, if writing.
    pub to: Option<Tier>,
    /// Byte offset in the source.
    pub offset: u64,
    /// Number of bytes.
    pub length: u64,
}

/// Result of an I/O operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoResult {
    /// The request that produced this result.
    pub request: IoRequest,
    /// Number of bytes actually transferred.
    pub bytes_transferred: u64,
    /// Wall-clock duration of the operation.
    pub duration_nanos: u64,
    /// Whether the operation succeeded.
    pub success: bool,
}
