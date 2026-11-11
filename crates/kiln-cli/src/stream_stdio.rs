//! Structured, private JSONL worker for the local API's streamed MoE child.
//! The parent `kiln stream` validates the GGUF and creates the stream index.
use std::io::{self, BufRead, Read, Write};

const MAX_REQUEST_LINE: usize = 262_144;

fn next_request_line<R: BufRead>(reader: &mut R) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    let count = reader.take((MAX_REQUEST_LINE as u64) + 1)
        .read_until(b'\n', &mut bytes).map_err(|e| format!("stream IPC input error: {e}"))?;
    if count == 0 { return Ok(None); }
    if count > MAX_REQUEST_LINE || bytes.last() != Some(&b'\n') {
        return Err("stream IPC message exceeds 256 KiB or is truncated".into());
    }
    String::from_utf8(bytes).map(Some)
        .map_err(|_| "stream IPC request is not valid UTF-8".into())
}


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
    let stdin = io::stdin();
    let mut input = stdin.lock();
    while let Some(line) = next_request_line(&mut input)? {
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

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn request_limit_is_enforced_before_allocating_the_entire_line() {
        let mut valid = BufReader::new(Cursor::new(b"{\"prompt\":\"hello\"}\n".to_vec()));
        assert_eq!(next_request_line(&mut valid).unwrap().unwrap(), "{\"prompt\":\"hello\"}\n");
        assert!(next_request_line(&mut valid).unwrap().is_none());
        let mut truncated = BufReader::new(Cursor::new(b"{}".to_vec()));
        assert!(next_request_line(&mut truncated).is_err());
        let mut large = BufReader::new(Cursor::new(vec![b'a'; MAX_REQUEST_LINE + 500]));
        assert!(next_request_line(&mut large).is_err());
    }
}
