//! KILN I/O Abstraction.
//!
//! Provides a unified async read and write API across platforms. The
//! implementation for each platform is selected at compile time:
//!
//!   Linux:   io_uring with O_DIRECT
//!   Windows: IOCP with overlapped I/O
//!   macOS:   kqueue
//!
//! A synchronous fallback is provided for platforms and environments where
//! none of the async backends are available. The fallback is always
//! compiled so that tests run everywhere.
//!
//! See docs/architecture.md, Layer 2, for the tier model that this crate
//! serves.

pub mod request;
pub mod source;
pub mod fallback;

pub use request::{IoRequest, IoResult, IoOp};
pub use source::{IoSource, BlockSize};
pub use fallback::SyncIoSource;
