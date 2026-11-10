//! Structured, private JSONL worker for the local API's streamed MoE child.
//! The parent `kiln stream` validates the GGUF and creates the stream index.
use std::io::{self, BufRead, Write};

use kiln_runtime::{LlamaCppBackend, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};

fn emit(event: serde_json::Value) -> Result<(), String> {
    let mut out = io::stdout().lock();
    writeln!(out, "{event}").and_then(|_| out.flush()).map_err(|e| e.to_string())
}

pub fn run(model: &str, lazy_moe_experts: bool) -> Result<(), String> {
    let profile = SystemProfile::load_or_probe();
    let mut dispatcher = RuntimeDispatcher::new();
    dispatcher.register(Box::new(LlamaCppBackend::new()));
    dispatcher.dispatch(RuntimeOp::Load {
        path: std::path::PathBuf::from(model),
        n_ctx: profile.n_ctx,
        n_batch: profile.n_batch,
        n_ubatch: profile.n_batch.min(512),
        n_threads: profile.n_threads,
        lazy_moe_experts,
    }).map_err(|e| format!("native stream worker load failed: {e}"))?;
    let eos = match dispatcher.dispatch(RuntimeOp::EosToken).map_err(|e| e.to_string())? {
        RuntimeResult::Int(token) => token,
        _ => return Err("native worker returned invalid EOS token".into()),
    };
    emit(serde_json::json!({"event":"ready"}))?;
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.len() > 262_144 { return Err("stream IPC message too large".into()); }
        let req: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid IPC JSON: {e}"))?;
        let prompt = req.get("prompt").and_then(|x| x.as_str())
            .ok_or("stream IPC requires a prompt")?;
        let tokens = req.get("max_tokens").and_then(|x| x.as_u64()).unwrap_or(128);
        if prompt.is_empty() || prompt.len() > 131_072 || tokens == 0 || tokens > 4096 {
            return Err("invalid stream IPC prompt length or token limit".into());
        }
        dispatcher.dispatch(RuntimeOp::Reset).map_err(|e| format!("stream reset: {e}"))?;
        let encoded = match dispatcher.dispatch(RuntimeOp::Tokenize { text: prompt.to_string() })
            .map_err(|e| e.to_string())? {
            RuntimeResult::Tokens(value) if !value.is_empty() => value,
            _ => return Err("empty or invalid tokenization".into()),
        };
        let mut token = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: encoded })
            .map_err(|e| format!("stream prefill failed: {e}"))? {
            RuntimeResult::Int(value) => value,
            _ => return Err("stream prefill produced invalid token".into()),
        };
        let mut pending = Vec::<u8>::new();
        for n in 0..tokens {
            if token == eos { break; }
            let raw = match dispatcher.dispatch(RuntimeOp::TokenToStr { token })
                .map_err(|e| format!("stream token decode failed: {e}"))? {
                RuntimeResult::Bytes(bytes) => bytes,
                RuntimeResult::Str(text) => text.into_bytes(),
                _ => return Err("stream decoder returned invalid bytes".into()),
            };
            pending.extend(raw);
            if let Ok(piece) = std::str::from_utf8(&pending) {
                if !piece.is_empty() { emit(serde_json::json!({"event":"chunk","text":piece}))?; }
                pending.clear();
            } else if pending.len() > 16 {
                return Err("invalid native token UTF-8 sequence".into());
            }
            if n+1 < tokens {
                token = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: vec![token] })
                    .map_err(|e| format!("stream decode failed: {e}"))? {
                    RuntimeResult::Int(value) => value,
                    _ => return Err("stream decode produced invalid token".into()),
                };
            }
        }
        if !pending.is_empty() {
            emit(serde_json::json!({"event":"chunk","text":String::from_utf8_lossy(&pending)}))?;
        }
        emit(serde_json::json!({"event":"done"}))?;
    }
    dispatcher.dispatch(RuntimeOp::Unload).map_err(|e| e.to_string())?;
    Ok(())
}
