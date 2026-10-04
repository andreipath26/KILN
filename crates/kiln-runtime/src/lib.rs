//! KILN runtime. A thin wrapper around llama.cpp's C API.
//! See roadmap v7.0 and docs/runtime-loading-design.md.

pub mod ffi;
pub mod loader;

pub use ffi::{LlamaContext, LlamaError};
pub use loader::{find_existing, Library, LoadError};
