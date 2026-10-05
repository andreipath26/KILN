//! KILN runtime. A thin wrapper around llama.cpp's C API.
//! See roadmap v7.0 and docs/runtime-loading-design.md.

pub mod ffi;
pub mod loader;

pub use ffi::{LlamaContext, LlamaError};
pub use loader::{find_existing, Library, LoadError};
pub mod profile;
pub use profile::{SystemProfile, Tier};
pub mod runtime_dispatch;
pub mod backends;
pub use runtime_dispatch::{RuntimeBackend, RuntimeDispatcher, RuntimeError, RuntimeOp, RuntimeResult};
pub use backends::llama_cpp::LlamaCppBackend;
pub mod sanity;
pub use sanity::{run_canaries, Canary, SanityError, Watchdog, WatchdogVerdict, CANARIES};
