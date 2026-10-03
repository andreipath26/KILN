//! Request and response types matching the Ollama API contract.

use serde::{Deserialize, Serialize};

/// Request to POST /api/generate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
}

/// Response from POST /api/generate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
}

/// Request to POST /api/chat.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
}

/// One message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Response from POST /api/chat.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponse {
    pub model: String,
    pub message: ChatMessage,
    pub done: bool,
}

/// Response from GET /api/tags.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TagsResponse {
    pub models: Vec<ModelInfo>,
}

/// Information about one model.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// Request to POST /api/pull.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PullRequest {
    pub name: String,
    #[serde(default)]
    pub stream: Option<bool>,
}

/// Response from GET /api/version.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VersionResponse {
    pub version: String,
}
