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
pub mod range_reader;
pub mod expert_blob;
pub mod cached_expert_blob;

pub use request::{IoRequest, IoResult, IoOp};
pub use source::{IoSource, BlockSize};
pub use fallback::SyncIoSource;
pub use range_reader::{FileRangeReader, RangeReadError};
pub use expert_blob::{ExpertBlobError, ExpertBlobReader};
pub use cached_expert_blob::{CachedExpertBlobReader, ExpertAccessTraceSnapshot};

pub mod expert_sidecar;
pub use expert_sidecar::{write_expert_sidecar, verify_expert_sidecar, SidecarIntegrityManifest, SidecarPackError, SIDECAR_ALIGNMENT};

pub use expert_sidecar::{write_component_expert_sidecar, verify_component_expert_sidecar, PackedExpertComponentMap, PackedExpertComponent};

pub mod verified_component_reader;
pub use verified_component_reader::{VerifiedExpertComponentReader, SharedTensorComponent, ComponentReadError};

pub use expert_sidecar::verify_component_sidecar_against_source;

pub mod guarded_expert_loader;
pub use guarded_expert_loader::{GuardedExpertLoader, RoutedExpertLoad};

pub mod async_expert_prefetch;
pub use async_expert_prefetch::{AsyncExpertPrefetch, AsyncExpertStats, ExpertReadCompletion, SubmitResult};

pub mod async_guarded_coordinator;
pub use async_guarded_coordinator::AsyncGuardedExpertCoordinator;
