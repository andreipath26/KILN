//! Route handlers for the Ollama-compatible API.

use axum::{http::StatusCode, Json};
use serde_json::{json, Value};

use crate::types::{GenerateRequest, TagsResponse, VersionResponse};

/// POST /api/generate.
///
/// Generation is deliberately reported as unsupported until the API is wired to
/// a real, loaded runtime session. Never return a successful placeholder answer.
pub async fn generate(
    Json(_req): Json<GenerateRequest>,
) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Model generation is not wired to the runtime yet."
        })),
    )
}

/// POST /api/chat.
///
/// Chat is deliberately reported as unsupported until the API is wired to a
/// real, loaded runtime session. Never return a successful placeholder answer.
pub async fn chat(
    Json(_req): Json<crate::types::ChatRequest>,
) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Chat generation is not wired to the runtime yet."
        })),
    )
}

/// GET /api/tags.
///
/// No model registry is connected to the API yet, so the current inventory is
/// empty. Do not invent model entries or claim that models are loaded.
pub async fn tags() -> Json<TagsResponse> {
    Json(TagsResponse { models: vec![] })
}

/// GET /api/version.
pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest, GenerateRequest};

    #[tokio::test]
    async fn generate_does_not_report_placeholder_as_success() {
        let request = GenerateRequest {
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            stream: Some(false),
            options: None,
        };

        let (status, Json(body)) = generate(Json(request)).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            body["error"],
            "Model generation is not wired to the runtime yet."
        );
    }

    #[tokio::test]
    async fn chat_does_not_report_placeholder_as_success() {
        let request = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            stream: Some(false),
        };

        let (status, Json(body)) = chat(Json(request)).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            body["error"],
            "Chat generation is not wired to the runtime yet."
        );
    }

    #[tokio::test]
    async fn tags_reports_an_empty_inventory_until_registry_is_connected() {
        let Json(body) = tags().await;
        assert!(body.models.is_empty());
    }
}
