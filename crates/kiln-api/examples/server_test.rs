//! Starts the KILN API server. Run with:
//!   cargo run --example server_test -p kiln-api
//! Then in another terminal:
//!   curl http://127.0.0.1:11435/api/version
//!   curl http://127.0.0.1:11435/api/tags

use kiln_api::serve;

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("KILN_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11435);
    println!("starting KILN API on 127.0.0.1:{}", port);
    if let Err(e) = serve("127.0.0.1", port).await {
        eprintln!("server error: {}", e);
        std::process::exit(1);
    }
}
