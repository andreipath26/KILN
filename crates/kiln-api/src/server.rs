//! HTTP server setup.

use axum::http::header;
use axum::response::Html;
use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;

use crate::{routes, runtime::ApiState};

/// Build the API router without a configured model, useful for health/version
/// routes and tests. Inference endpoints fail closed in this mode.
pub fn build_router() -> Router {
    build_router_with_state(ApiState::unconfigured())
}

fn build_router_with_state(state: ApiState) -> Router {
    Router::new()
        .route("/", get(ui_index))
        .route("/ui.css", get(ui_css))
        .route("/ui.js", get(ui_js))
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        .route("/api/version", get(routes::version))
        .route("/api/ui/capabilities", get(routes::ui_capabilities))
        .with_state(state)
}

async fn ui_index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn ui_css() -> impl axum::response::IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], include_str!("../static/ui.css"))
}

async fn ui_js() -> impl axum::response::IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("../static/ui.js"))
}

/// Start the server on the given host and port. If KILN_MODEL_PATH is set, load
/// that GGUF model once at startup and expose it to the inference endpoints.
pub async fn serve(host: &str, port: u16) -> std::io::Result<()> {
    let state = ApiState::from_env()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("KILN API listening on {}", addr);
    axum::serve(listener, build_router_with_state(state))
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
