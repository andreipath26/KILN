//! Route handlers for the Ollama-compatible API.

use axum::Json;

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, GenerateRequest, GenerateResponse,
    TagsResponse, VersionResponse,
};

/// POST /api/generate.
pub async fn generate(Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    // Placeholder. The real implementation routes through kiln-core.
    Json(GenerateResponse {
        model: req.model,
        response: "KILN placeholder response. Runtime integration lands in Phase 1.".to_string(),
        done: true,
        context: None,
        total_duration: None,
    })
}

/// POST /api/chat.
pub async fn chat(Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    Json(ChatResponse {
        model: req.model,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "KILN placeholder response. Runtime integration lands in Phase 1.".to_string(),
        },
        done: true,
    })
}

/// GET /api/tags.
pub async fn tags() -> Json<TagsResponse> {
    Json(TagsResponse { models: vec![] })
}

/// GET /api/version.
pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
