//! The IoSource trait.

use crate::request::{IoRequest, IoResult};

/// Sector alignment hint for O_DIRECT on some platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSize {
    /// Bytes are not block-aligned.
    Unaligned,
    /// 512-byte sectors.
    B512,
    /// 4096-byte sectors. The common default on NVMe.
    B4096,
}

/// Any I/O provider implements this trait. The trait is deliberately small:
/// submit a batch of requests, receive results in the same order. Batching
/// is how io_uring and IOCP achieve their throughput.
pub trait IoSource: Send + Sync {
    /// Submit a batch of I/O requests. Returns results in the same order.
    fn submit(&mut self, requests: &[IoRequest]) -> Vec<IoResult>;

    /// The alignment this source requires for its reads and writes.
    fn block_size(&self) -> BlockSize;

    /// The number of requests that can be in flight at once.
    fn queue_depth(&self) -> u32;

    /// Flush any pending writes.
    fn flush(&mut self) -> std::io::Result<()>;
}
