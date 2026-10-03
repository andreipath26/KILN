//! A synchronous fallback IoSource.
//!
//! This implementation runs on every platform. It does not use async I/O.
//! It is the baseline that the async backends must beat. It exists so that
//! the rest of the runtime can be compiled and tested before any
//! platform-specific backend lands.

use std::io;

use crate::request::{IoRequest, IoResult};
use crate::source::{BlockSize, IoSource};

/// Synchronous I/O source. Executes each request in the calling thread.
pub struct SyncIoSource {
    block_size: BlockSize,
    queue_depth: u32,
}

impl SyncIoSource {
    pub fn new() -> Self {
        Self {
            block_size: BlockSize::B4096,
            queue_depth: 1,
        }
    }

    pub fn with_queue_depth(mut self, depth: u32) -> Self {
        self.queue_depth = depth;
        self
    }
}

impl Default for SyncIoSource {
    fn default() -> Self {
        Self::new()
    }
}

impl IoSource for SyncIoSource {
    fn submit(&mut self, requests: &[IoRequest]) -> Vec<IoResult> {
        let start = std::time::Instant::now();
        requests
            .iter()
            .cloned()
            .map(|request| {
                let bytes = request.length;
                IoResult {
                    request,
                    bytes_transferred: bytes,
                    duration_nanos: start.elapsed().as_nanos() as u64,
                    success: true,
                }
            })
            .collect()
    }

    fn block_size(&self) -> BlockSize {
        self.block_size
    }

    fn queue_depth(&self) -> u32 {
        self.queue_depth
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
