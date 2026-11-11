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

/// The final prefill token produces the first output token without adding
/// that sampled token to KV. Subsequent samples need one KV position each.
/// Cap generation before it reaches the context boundary mid-stream.
fn available_generation_steps(
    prompt_tokens: usize, context_tokens: usize, requested: usize
) -> Result<usize, String> {
    if prompt_tokens == 0 { return Err("prompt produced no tokens".into()); }
    if context_tokens == 0 || prompt_tokens > context_tokens {
        return Err(format!(
            "prompt requires {prompt_tokens} tokens but the active model context holds {context_tokens}; shorten the chat history or increase the context limit"
        ));
    }
    if requested == 0 { return Err("requested token limit must be positive".into()); }
    Ok(requested.min(context_tokens - prompt_tokens + 1))
}

/// Assemble complete UTF-8 without changing token bytes. A partial
/// multi-byte codepoint may span multiple token pieces; malformed bytes may
/// not be silently replaced because doing so changes generated text.
fn parse_request_token_limit(req: &serde_json::Value) -> Result<u64, String> {
    let tokens = match req.get("max_tokens") {
        None => 128,
        Some(value) => value.as_u64()
            .ok_or("stream IPC max_tokens must be a positive integer")?,
    };
    if !(1..=4096).contains(&tokens) {
        return Err("stream IPC max_tokens must be between 1 and 4096".into());
    }
    Ok(tokens)
}

fn push_token_bytes(pending: &mut Vec<u8>, raw: &[u8]) -> Result<Option<String>, String> {
    pending.extend_from_slice(raw);
    match std::str::from_utf8(pending) {
        Ok(text) => {
            let completed = text.to_owned();
            pending.clear();
            Ok((!completed.is_empty()).then_some(completed))
        }
        Err(error) if error.error_len().is_some() =>
            Err("native tokenizer emitted invalid UTF-8 bytes".into()),
        Err(_) if pending.len() > 4 =>
            Err("native tokenizer emitted an overlong incomplete UTF-8 sequence".into()),
        Err(_) => Ok(None),
    }
}

fn finish_token_bytes(pending: &[u8]) -> Result<(), String> {
    if pending.is_empty() { Ok(()) }
    else { Err("native tokenizer ended with an incomplete UTF-8 sequence".into()) }
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
        let tokens = parse_request_token_limit(&req)?;
        if prompt.is_empty() || prompt.len() > 131_072 {
            return Err("invalid stream IPC prompt length".into());
        }
        dispatcher.dispatch(RuntimeOp::Reset).map_err(|e| format!("stream reset: {e}"))?;
        let encoded = match dispatcher.dispatch(RuntimeOp::Tokenize { text: prompt.to_string() })
            .map_err(|e| e.to_string())? {
            RuntimeResult::Tokens(value) if !value.is_empty() => value,
            _ => return Err("empty or invalid tokenization".into()),
        };
        let max_steps = match available_generation_steps(
            encoded.len(), profile.n_ctx as usize, tokens as usize
        ) {
            Ok(value) => value,
            Err(message) => {
                // This is a recoverable request rejection, not a broken
                // native worker. Signal the API without losing loaded weights.
                emit(serde_json::json!({"event":"reject","message":message}))?;
                emit(serde_json::json!({"event":"done"}))?;
                continue;
            }
        };
        let mut token = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: encoded })
            .map_err(|e| format!("stream prefill failed: {e}"))? {
            RuntimeResult::Int(value) => value,
            _ => return Err("stream prefill produced invalid token".into()),
        };
        let mut pending = Vec::<u8>::new();
        for n in 0..max_steps as u64 {
            if token == eos { break; }
            let raw = match dispatcher.dispatch(RuntimeOp::TokenToStr { token })
                .map_err(|e| format!("stream token decode failed: {e}"))? {
                RuntimeResult::Bytes(bytes) => bytes,
                RuntimeResult::Str(text) => text.into_bytes(),
                _ => return Err("stream decoder returned invalid bytes".into()),
            };
            if let Some(piece) = push_token_bytes(&mut pending, &raw)? {
                emit(serde_json::json!({"event":"chunk","text":piece}))?;
            }
            if n+1 < max_steps as u64 {
                token = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: vec![token] })
                    .map_err(|e| format!("stream decode failed: {e}"))? {
                    RuntimeResult::Int(value) => value,
                    _ => return Err("stream decode produced invalid token".into()),
                };
            }
        }
        finish_token_bytes(&pending)?;
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
    fn token_utf8_assembler_preserves_split_codepoints_without_replacement() {
        let mut pending = Vec::new();
        assert_eq!(push_token_bytes(&mut pending, &[0xE2]).unwrap(), None);
        assert_eq!(push_token_bytes(&mut pending, &[0x82]).unwrap(), None);
        assert_eq!(push_token_bytes(&mut pending, &[0xAC]).unwrap(), Some("€".into()));
        finish_token_bytes(&pending).unwrap();
        assert_eq!(push_token_bytes(&mut pending, b"ok").unwrap(), Some("ok".into()));
        assert!(pending.is_empty());
    }

    #[test]
    fn token_utf8_assembler_rejects_invalid_and_truncated_output() {
        let mut invalid = Vec::new();
        assert!(push_token_bytes(&mut invalid, &[0xff]).unwrap_err().contains("invalid UTF-8"));
        let mut incomplete = Vec::new();
        assert_eq!(push_token_bytes(&mut incomplete, &[0xF0, 0x9F]).unwrap(), None);
        assert!(finish_token_bytes(&incomplete).unwrap_err().contains("incomplete UTF-8"));
        assert!(push_token_bytes(&mut incomplete, &[0xFF]).is_err());
    }

    #[test]
    fn invalid_token_limit_is_not_silently_defaulted() {
        assert_eq!(parse_request_token_limit(&serde_json::json!({})).unwrap(), 128);
        assert_eq!(parse_request_token_limit(&serde_json::json!({"max_tokens":8})).unwrap(), 8);
        for value in [serde_json::json!("8"), serde_json::json!(-1),
                      serde_json::json!(null), serde_json::json!(0),
                      serde_json::json!(4097), serde_json::json!(true)] {
            assert!(parse_request_token_limit(&serde_json::json!({"max_tokens":value})).is_err());
        }
    }

    #[test]
    fn prefill_budget_caps_generation_before_running_off_context() {
        assert_eq!(available_generation_steps(1,2048,4096).unwrap(),2048);
        assert_eq!(available_generation_steps(2048,2048,128).unwrap(),1);
        assert_eq!(available_generation_steps(2047,2048,128).unwrap(),2);
        assert_eq!(available_generation_steps(8,2048,64).unwrap(),64);
        assert!(available_generation_steps(2049,2048,64).unwrap_err().contains("shorten"));
        assert!(available_generation_steps(0,2048,64).is_err());
        assert!(available_generation_steps(1,0,64).is_err());
        assert!(available_generation_steps(1,2048,0).is_err());
    }

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
