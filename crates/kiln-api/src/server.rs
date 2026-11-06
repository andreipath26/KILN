//! HTTP server setup.

use axum::http::{header, HeaderName, HeaderValue};
use axum::middleware::{self, Next};
use axum::http::Request;
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
        .route("/stream.js", get(ui_stream_js))
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        .route("/api/version", get(routes::version))
        .route("/api/ui/capabilities", get(routes::ui_capabilities))
        .route("/api/ui/models", get(routes::ui_models))
        .route("/api/ui/models/search", get(routes::ui_model_search))
        .route("/api/ui/models/pull", post(routes::ui_model_pull))
        .route("/api/ui/models/activate", post(routes::ui_model_activate))
        .layer(middleware::from_fn(security_headers))
        .with_state(state)
}

async fn security_headers(request: Request<axum::body::Body>, next: Next<axum::body::Body>) -> axum::response::Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(HeaderName::from_static("content-security-policy"), HeaderValue::from_static(
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ));
    headers.insert(HeaderName::from_static("x-content-type-options"), HeaderValue::from_static("nosniff"));
    headers.insert(HeaderName::from_static("x-frame-options"), HeaderValue::from_static("DENY"));
    headers.insert(HeaderName::from_static("referrer-policy"), HeaderValue::from_static("no-referrer"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
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

async fn ui_stream_js() -> impl axum::response::IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("../static/stream.js"))
}

/// Start the server on the given host and port. If KILN_MODEL_PATH is set, load
/// that GGUF model once at startup and expose it to the inference endpoints.
pub async fn serve(host: &str, port: u16) -> std::io::Result<()> {
    let ip = host.parse::<std::net::IpAddr>().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput,
            "KILN API host must be an explicit loopback IP address; remote access is not supported without authentication")
    })?;
    if !ip.is_loopback() {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied,
            "KILN API only supports loopback binding; remote access is not supported without authentication"));
    }
    let state = ApiState::from_env()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let addr = std::net::SocketAddr::new(ip, port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("KILN API listening on {}", addr);
    axum::serve(listener, build_router_with_state(state))
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
