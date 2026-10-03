//! HTTP server setup.

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;

use crate::routes;

/// Build the axum router with all endpoints.
pub fn build_router() -> Router {
    Router::new()
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        .route("/api/version", get(routes::version))
}

/// Start the server on the given host and port. Blocks until shutdown.
pub async fn serve(host: &str, port: u16) -> std::io::Result<()> {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("KILN API listening on {}", addr);
    axum::serve(listener, build_router())
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
