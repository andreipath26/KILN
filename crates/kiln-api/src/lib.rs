//! KILN API.
//!
//! Ollama-compatible REST server. See docs/architecture.md, Layer 6, and
//! Section 6 of the Master File for the contract.

pub mod types;
pub mod routes;
pub mod server;

pub use types::{
    GenerateRequest, GenerateResponse, ChatRequest, ChatResponse,
    TagsResponse, ModelInfo, PullRequest, VersionResponse,
};
pub use server::{build_router, serve};
