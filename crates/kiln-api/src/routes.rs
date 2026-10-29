//! Route handlers for the Ollama-compatible API.

use std::time::Instant;

use axum::{extract::State, http::StatusCode, Json};
use kiln_core::chat::{Sampler, SamplingStrategy};
use kiln_runtime::{RuntimeOp, RuntimeResult};
use serde_json::{json, Value};

use crate::{
    runtime::ApiState,
    types::{
        ChatMessage, ChatRequest, ChatResponse, GenerateRequest, GenerateResponse, ModelInfo,
        TagsResponse, VersionResponse,
    },
};

type ApiError = (StatusCode, String);
type ErrorResponse = (StatusCode, Json<Value>);

fn error_response((status, message): ApiError) -> ErrorResponse {
    (status, Json(json!({ "error": message })))
}

fn max_tokens_from_options(options: Option<&Value>) -> Result<usize, ApiError> {
    const DEFAULT_MAX_TOKENS: usize = 128;
    const MAX_ALLOWED_TOKENS: usize = 4096;

    let Some(options) = options else {
        return Ok(DEFAULT_MAX_TOKENS);
    };
    let Some(object) = options.as_object() else {
        return Err((StatusCode::BAD_REQUEST, "options must be a JSON object".into()));
    };

    let unsupported: Vec<&str> = object
        .keys()
        .filter(|key| key.as_str() != "num_predict")
        .map(String::as_str)
        .collect();
    if !unsupported.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported generation option(s): {}", unsupported.join(", ")),
        ));
    }

    let Some(value) = object.get("num_predict") else {
        return Ok(DEFAULT_MAX_TOKENS);
    };
    let Some(value) = value.as_u64() else {
        return Err((
            StatusCode::BAD_REQUEST,
            "options.num_predict must be a positive integer".into(),
        ));
    };
    if value == 0 || value > MAX_ALLOWED_TOKENS as u64 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("options.num_predict must be between 1 and {}", MAX_ALLOWED_TOKENS),
        ));
    }
    Ok(value as usize)
}

/// Run one greedy completion against the configured runtime session.
fn infer(
    state: &ApiState,
    requested_model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<(String, String), ApiError> {
    if prompt.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt must not be empty".into()));
    }

    let mut runtime = state.runtime.lock().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, "runtime state lock is poisoned".into())
    })?;
    let session = runtime.as_mut().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is configured; set KILN_MODEL_PATH before starting the server".into(),
        )
    })?;

    if requested_model != session.model_name {
        return Err((
            StatusCode::NOT_FOUND,
            format!("model '{}' is not loaded; loaded model is '{}'", requested_model, session.model_name),
        ));
    }

    session.dispatcher.dispatch(RuntimeOp::Reset)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("session reset failed: {}", e)))?;
    let prompt_tokens = match session.dispatcher.dispatch(RuntimeOp::Tokenize {
        text: prompt.to_string(),
    }).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenization failed: {}", e)))? {
        RuntimeResult::Tokens(tokens) => tokens,
        _ => return Err((StatusCode::INTERNAL_SERVER_ERROR, "runtime returned an unexpected tokenization result".into())),
    };
    if prompt_tokens.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt tokenized to no tokens".into()));
    }

    let mut logits = match session.dispatcher.dispatch(RuntimeOp::Forward { tokens: prompt_tokens })
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("prompt evaluation failed: {}", e)))? {
        RuntimeResult::Logits(logits) => logits,
        _ => return Err((StatusCode::INTERNAL_SERVER_ERROR, "runtime returned an unexpected prompt result".into())),
    };

    let eos = match session.dispatcher.dispatch(RuntimeOp::EosToken)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("EOS lookup failed: {}", e)))? {
        RuntimeResult::Int(token) => token,
        _ => return Err((StatusCode::INTERNAL_SERVER_ERROR, "runtime returned an unexpected EOS-token result".into())),
    };
    let mut sampler = Sampler::new(SamplingStrategy::Greedy, 42);
    let mut output = Vec::new();

    for step in 0..max_tokens {
        let token = sampler.sample(&logits)
            .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, "sampler returned no token".into()))? as i32;
        if token == eos {
            break;
        }
        match session.dispatcher.dispatch(RuntimeOp::TokenToStr { token })
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("token decoding failed: {}", e)))? {
            RuntimeResult::Bytes(bytes) => output.extend_from_slice(&bytes),
            RuntimeResult::Str(text) => output.extend_from_slice(text.as_bytes()),
            _ => return Err((StatusCode::INTERNAL_SERVER_ERROR, "runtime returned an unexpected token-string result".into())),
        }

        // No need to evaluate another token after the final requested output token.
        if step + 1 < max_tokens {
            logits = match session.dispatcher.dispatch(RuntimeOp::Forward { tokens: vec![token] })
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("decode step failed: {}", e)))? {
                RuntimeResult::Logits(logits) => logits,
                _ => return Err((StatusCode::INTERNAL_SERVER_ERROR, "runtime returned an unexpected decode result".into())),
            };
        }
    }

    Ok((session.model_name.clone(), String::from_utf8_lossy(&output).into_owned()))
}

/// POST /api/generate. Streaming and generation options other than num_predict
/// are rejected rather than silently pretending to support them.
pub async fn generate(
    State(state): State<ApiState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, ErrorResponse> {
    if request.stream.unwrap_or(false) {
        return Err(error_response((StatusCode::NOT_IMPLEMENTED, "streaming responses are not implemented yet".into())));
    }
    let max_tokens = max_tokens_from_options(request.options.as_ref()).map_err(error_response)?;
    let started = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        infer(&state, &request.model, &request.prompt, max_tokens)
    }).await.map_err(|e| error_response((StatusCode::INTERNAL_SERVER_ERROR, format!("inference task failed: {}", e))))?;
    let (model, response) = result.map_err(error_response)?;
    Ok(Json(GenerateResponse {
        model,
        response,
        done: true,
        context: None,
        total_duration: Some(started.elapsed().as_nanos().min(u64::MAX as u128) as u64),
    }))
}

/// POST /api/chat. Messages are rendered as explicit role-prefixed text because
/// model-specific chat-template application is not exposed by the current shim.
pub async fn chat(
    State(state): State<ApiState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ErrorResponse> {
    if request.stream.unwrap_or(false) {
        return Err(error_response((StatusCode::NOT_IMPLEMENTED, "streaming responses are not implemented yet".into())));
    }
    if request.messages.is_empty() {
        return Err(error_response((StatusCode::BAD_REQUEST, "messages must not be empty".into())));
    }

    let mut prompt = String::new();
    for ChatMessage { role, content } in &request.messages {
        let role = role.trim();
        if role.is_empty() {
            return Err(error_response((StatusCode::BAD_REQUEST, "message role must not be empty".into())));
        }
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(content);
        prompt.push('\n');
    }
    prompt.push_str("assistant: ");

    let started = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        infer(&state, &request.model, &prompt, 128)
    }).await.map_err(|e| error_response((StatusCode::INTERNAL_SERVER_ERROR, format!("inference task failed: {}", e))))?;
    let (model, content) = result.map_err(error_response)?;
    let _total_duration_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    Ok(Json(ChatResponse {
        model,
        message: ChatMessage { role: "assistant".into(), content },
        done: true,
    }))
}

/// GET /api/tags. The configured model is reported when one is loaded.
pub async fn tags(State(state): State<ApiState>) -> Json<TagsResponse> {
    let models = match state.runtime.lock() {
        Ok(runtime) => runtime.as_ref().map(|session| {
            let size = std::fs::metadata(&session.model_path).map(|metadata| metadata.len()).unwrap_or(0);
            vec![ModelInfo {
                name: session.model_name.clone(),
                size,
                digest: String::new(),
                modified_at: String::new(),
            }]
        }).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    Json(TagsResponse { models })
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

    #[test]
    fn options_reject_unsupported_values_instead_of_ignoring_them() {
        assert_eq!(max_tokens_from_options(Some(&json!({"temperature": 0.7}))).unwrap_err().0, StatusCode::BAD_REQUEST);
        assert_eq!(max_tokens_from_options(Some(&json!({"num_predict": 0}))).unwrap_err().0, StatusCode::BAD_REQUEST);
        assert_eq!(max_tokens_from_options(Some(&json!({"num_predict": 12}))).unwrap(), 12);
    }

    #[tokio::test]
    async fn generate_fails_closed_when_no_model_is_configured() {
        let request = GenerateRequest {
            model: "test-model".into(),
            prompt: "hello".into(),
            stream: Some(false),
            options: None,
        };
        let error = generate(State(ApiState::unconfigured()), Json(request)).await.unwrap_err();
        assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_fails_closed_when_no_model_is_configured() {
        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hello".into() }],
            stream: Some(false),
        };
        let error = chat(State(ApiState::unconfigured()), Json(request)).await.unwrap_err();
        assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn tags_is_empty_when_no_model_is_configured() {
        let Json(response) = tags(State(ApiState::unconfigured())).await;
        assert!(response.models.is_empty());
    }
}
