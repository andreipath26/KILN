//! Isolated native expert streaming worker used only by the loopback UI.
//! Runs the verified `kiln stream` launcher instead of mutating process-wide
//! streaming environment variables inside the API server.
use std::io::{BufRead, BufReader, Read, Write};
use std::time::{Duration, Instant};

const MAX_IPC_LINE: usize = 128 * 1024;
// Must match the child JSONL reader's 256 KiB cap, including the newline.
const MAX_REQUEST_LINE: usize = 262_144;
const MAX_HANDSHAKE_LINE: usize = 4096;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const MAX_NATIVE_CACHE_MIB: u32 = 256;
const DEFAULT_STARTUP_TIMEOUT_SECONDS: u64 = 300;

fn startup_timeout_seconds(value: Option<&str>) -> u64 {
    value.and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_STARTUP_TIMEOUT_SECONDS).clamp(30, 900)
}

fn terminate_child_group(child: &mut Child) {
    // The launcher and its nested compute child have their own group.
    #[cfg(unix)]
    {
        unsafe extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
        if let Ok(pid) = i32::try_from(child.id()) {
            unsafe { let _ = kill(-pid, 9); }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}



/// Expert cache is in addition to GGML scratch, KV, page cache and the model
/// loader. Derive a deliberately small allowance from *current resident
/// budget*, not installed physical RAM, so a pressured Tier 0 machine does
/// not accidentally allocate the same 64 MiB as an unconstrained server.
fn native_cache_mib(resident_budget_bytes: Option<u64>) -> u32 {
    // When headroom cannot be measured, use the established conservative 64
    // MiB default; the native runtime continues to enforce its own limit.
    match resident_budget_bytes {
        None => 64,
        Some(bytes) => (bytes / (64 * 1024 * 1024))
            .min(u64::from(MAX_NATIVE_CACHE_MIB)) as u32,
    }
}



/// Limit bytes *during* input, not after read_line has already allocated them.
/// EOF is distinguished from an unterminated/truncated JSONL response.
fn read_bounded_line<R: BufRead>(reader: &mut R, limit: usize) -> Result<Option<String>, String> {
    let mut out = Vec::new();
    let mut bounded = reader.take((limit as u64).saturating_add(1));
    let n = bounded.read_until(b'\n', &mut out)
        .map_err(|e| format!("stream IPC read failed: {e}"))?;
    if n == 0 { return Ok(None); }
    if n > limit || out.last() != Some(&b'\n') {
        return Err("stream IPC line exceeded its limit or was truncated".into());
    }
    String::from_utf8(out).map(Some).map_err(|_| "stream IPC text was not UTF-8".into())
}

use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// A complete preflight rejection is an input error, not a failed worker.
/// All framing, IPC and process failures remain gateway errors.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamInferError {
    Rejected(String),
    Worker(String),
}

impl std::fmt::Display for StreamInferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(message) | Self::Worker(message) => f.write_str(message),
        }
    }
}

/// Encoding can expand quotes, control characters and backslashes. Check the
/// actual UTF-8 JSONL frame before writing to the child to avoid poisoning a
/// healthy loaded session with an over-limit request.
fn encode_request(prompt: &str, max_tokens: usize) -> Result<String, StreamInferError> {
    let frame = serde_json::json!({"prompt":prompt,"max_tokens":max_tokens}).to_string();
    if frame.len().checked_add(1).is_none_or(|bytes| bytes > MAX_REQUEST_LINE) {
        return Err(StreamInferError::Rejected(
            "streamed prompt exceeds the native 256 KiB encoded request limit; shorten the prompt".into()));
    }
    Ok(frame)
}

pub struct StreamWorker {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    protocol_failed: bool,
    cache_mib: u32,
}

impl StreamWorker {
    pub fn start(path: &Path) -> Result<Self, String> {
        if !cfg!(target_os="linux") {
            return Err("native selected-expert CPU streaming requires Linux".into());
        }
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut command = Command::new(exe);
        // The launcher spawns a native compute worker. Keep both in one
        // process group so a stuck child can be terminated on model switch.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let capacity = native_cache_mib(crate::runtime::memory_headroom_bytes()
            .map(|(_, resident)| resident));
        tracing::info!(cache_mib = capacity, model = %path.display(),
            "starting bounded native MoE expert cache");
        let mut child = command
            .arg("stream").arg(path)
            .arg("--stdio-json").arg("--parallel-reads")
            .arg("--cache-mib").arg(capacity.to_string())
            .stdin(Stdio::piped()).stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("could not start native streaming worker: {e}"))?;
        let input = child.stdin.take().ok_or("stream worker stdin unavailable")?;
        let output = child.stdout.take().ok_or("stream worker stdout unavailable")?;
        // A real larger-than-RAM model may need nontrivial startup time, but
        // waiting forever inside the API activation lock is unacceptable.
        // Read on one dedicated thread so the launcher can be terminated
        // if it never produces its ready frame.
        let seconds = startup_timeout_seconds(std::env::var("KILN_STREAM_STARTUP_TIMEOUT_SECS")
            .ok().as_deref());
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut output = BufReader::new(output);
            let line = read_bounded_line(&mut output, MAX_HANDSHAKE_LINE);
            let _ = tx.send((output, line));
        });
        let (output, line) = match rx.recv_timeout(Duration::from_secs(seconds)) {
            Ok(value) => value,
            Err(error) => {
                terminate_child_group(&mut child);
                return Err(format!(
                    "native model loading did not complete within {seconds}s ({error}); worker and nested compute process terminated"
                ));
            }
        };
        let line = match line {
            Ok(Some(line)) => line,
            Ok(None) => {
                terminate_child_group(&mut child);
                return Err("native worker exited before reporting ready; inspect server logs for unsupported GGUF or memory exhaustion".into());
            }
            Err(error) => {
                terminate_child_group(&mut child);
                return Err(format!("native worker emitted invalid readiness frame: {error}"));
            }
        };
        let response: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                terminate_child_group(&mut child);
                return Err(format!("malformed native worker readiness JSON: {error}"));
            }
        };
        if response.get("event").and_then(|s| s.as_str()) != Some("ready") {
            terminate_child_group(&mut child);
            return Err("native worker failed to confirm model loading".into());
        }
        Ok(Self {
            child, input: Some(input), output, protocol_failed: false,
            cache_mib: capacity,
        })
    }

    pub fn cache_mib(&self) -> u32 { self.cache_mib }

    pub fn infer(&mut self, prompt: &str, max_tokens: usize,
                 on_chunk: &mut dyn FnMut(&str) -> bool) -> Result<String, String> {
        self.infer_detailed(prompt, max_tokens, on_chunk).map_err(|e| e.to_string())
    }

    pub fn infer_detailed(&mut self, prompt: &str, max_tokens: usize,
                          on_chunk: &mut dyn FnMut(&str) -> bool) -> Result<String, StreamInferError> {
        if self.protocol_failed {
            return Err(StreamInferError::Worker(
                "stream worker protocol failed during a previous request; reselect the model to start a fresh worker".into()));
        }
        if prompt.is_empty() || prompt.len() > 131_072 || max_tokens == 0 || max_tokens > 4096 {
            return Err(StreamInferError::Rejected(
                "streamed prompt or token count exceeds bounded IPC limits".into()));
        }
        let request_line = encode_request(prompt, max_tokens)?;
        match self.infer_framed(&request_line, on_chunk) {
            Ok(Ok(text)) => Ok(text),
            // A fully framed rejection does not corrupt the loaded model.
            Ok(Err(message)) => Err(StreamInferError::Rejected(message)),
            Err(message) => {
                // Unknown/truncated frames remain fatal to this session.
                self.protocol_failed = true;
                Err(StreamInferError::Worker(message))
            }
        }
    }

    fn infer_framed(&mut self, request_line: &str,
                    on_chunk: &mut dyn FnMut(&str) -> bool) -> Result<Result<String, String>, String> {
        let input = self.input.as_mut().ok_or("stream worker stdin has closed")?;
        writeln!(input,"{request_line}").and_then(|_| input.flush())
            .map_err(|e| format!("streamed model process is unavailable: {e}"))?;
        let mut combined = String::new();
        let mut rejection: Option<String> = None;
        let mut saw_chunk = false;
        let mut receiver_open = true;
        loop {
            let line = read_bounded_line(&mut self.output, MAX_IPC_LINE)?
                .ok_or("native streaming worker exited during generation; see server logs")?;
            let response: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("malformed streamed response: {e}"))?;
            match response.get("event").and_then(|x| x.as_str()) {
                Some("chunk") => {
                    if rejection.is_some() {
                        return Err("stream worker emitted text after a rejected request".into());
                    }
                    let piece = response.get("text").and_then(|x| x.as_str())
                        .ok_or("stream worker emitted invalid text")?;
                    saw_chunk = true;
                    if receiver_open {
                        combined.push_str(piece);
                        if combined.len() > 4 * 1024 * 1024 {
                            return Err("stream worker response exceeds safe limit".into());
                        }
                    }
                    if receiver_open && !on_chunk(piece) {
                        // The HTTP client disconnected. Keep reading until
                        // the worker's "done" event so its IPC stream stays
                        // aligned for the next request, but send no more data.
                        receiver_open = false;
                    }
                }
                Some("reject") => {
                    let message = response.get("message").and_then(|x| x.as_str())
                        .ok_or("stream worker emitted an invalid rejection")?;
                    if message.is_empty() || rejection.is_some() || saw_chunk {
                        return Err("stream worker emitted an out-of-order rejection".into());
                    }
                    rejection = Some(message.to_owned());
                }
                Some("done") => break,
                _ => return Err("stream worker emitted an unknown protocol event".into()),
            }
        }
        Ok(match rejection {
            Some(message) => Err(message),
            None => Ok(combined),
        })
    }
}

impl Drop for StreamWorker {
    fn drop(&mut self) {
        // Closing the pipe lets the real worker leave its input loop and
        // release the native model; its parent then verifies the read audit.
        self.input.take();
        // Allow the launcher to finish native audit after EOF, but never
        // block all model activation forever on a wedged worker.
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                _ => {
                    terminate_child_group(&mut self.child);
                    break;
                }
            }
        }
        let _ = self.child.wait();
    }
}

#[cfg(all(test, unix))]
mod recoverable_request_tests {
    use super::*;
    #[test]
    fn rejected_prefill_preserves_worker_and_next_reply_alignment() {
        // A simulated JSONL child verifies the framing contract without
        // loading any GGUF or claiming native parity.
        let script = concat!(
            r#"printf '%s\n' "#,
            r#"'{"event":"reject","message":"prompt exceeds context"}' "#,
            r#"'{"event":"done"}' "#,
            r#"'{"event":"chunk","text":"okay"}' "#,
            r#"'{"event":"done"}'; "#,
            "cat >/dev/null"
        );
        let mut child = Command::new("sh").arg("-c").arg(script)
            .stdin(Stdio::piped()).stdout(Stdio::piped())
            .spawn().unwrap();
        let mut worker = StreamWorker {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child, protocol_failed:false, cache_mib:0,
        };
        assert!(worker.infer("too long",8,&mut |_| true)
            .unwrap_err().contains("prompt exceeds context"));
        assert!(!worker.protocol_failed);
        assert_eq!(worker.infer("short",8,&mut |_| true).unwrap(),"okay");
        assert!(!worker.protocol_failed);
    }
}

#[cfg(test)]
mod request_encoding_tests {
    use super::*;

    #[test]
    fn serialized_request_limit_accounts_for_json_escaping() {
        let safe = encode_request("hello", 8).unwrap();
        assert!(safe.ends_with("}"));
        assert!(safe.len() + 1 <= MAX_REQUEST_LINE);
        let escaped = "\0".repeat(60_000);
        assert!(escaped.len() < 131_072);
        assert!(matches!(encode_request(&escaped, 8),
            Err(StreamInferError::Rejected(message)) if message.contains("encoded request limit")));
        // The rejection is local preflight; no native worker was contacted.
        let bounded = "a".repeat(131_072);
        assert!(encode_request(&bounded, 4096).is_ok());
    }
}

#[cfg(all(test, unix))]
mod rejection_classification_tests {
    use super::*;
    use std::io::BufReader;
    #[test]
    fn well_framed_rejection_is_not_a_gateway_failure() {
        let script = "printf '%s\\n' '{\"event\":\"reject\",\"message\":\"too long\"}' '{\"event\":\"done\"}'; cat >/dev/null";
        let mut child = Command::new("sh").arg("-c").arg(script)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let mut worker = StreamWorker {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child, protocol_failed: false, cache_mib: 0,
        };
        assert_eq!(worker.infer_detailed("prompt", 8, &mut |_| true),
            Err(StreamInferError::Rejected("too long".into())));
        assert!(!worker.protocol_failed);
    }
}

#[cfg(all(test, unix))]
mod invalid_frame_tests {
    use super::*;

    fn simulated_worker(script: &str) -> StreamWorker {
        let mut child = Command::new("sh").arg("-c").arg(script)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        StreamWorker {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child, protocol_failed: false, cache_mib: 0,
        }
    }

    #[test]
    fn empty_chunk_before_reject_poison_worker_instead_of_reporting_client_error() {
        let script = "printf '%s\\n' '{\"event\":\"chunk\",\"text\":\"\"}' '{\"event\":\"reject\",\"message\":\"bad\"}' '{\"event\":\"done\"}'; cat >/dev/null";
        let mut worker = simulated_worker(script);
        assert!(matches!(worker.infer_detailed("a", 2, &mut |_| true),
            Err(StreamInferError::Worker(message)) if message.contains("out-of-order")));
        assert!(worker.protocol_failed);
        assert!(matches!(worker.infer_detailed("b", 2, &mut |_| true), Err(StreamInferError::Worker(_))));
    }

    #[test]
    fn malformed_worker_frame_poisoning_is_not_recoverable() {
        let script = "printf '%s\\n' '{\"event\":\"unexpected\"}'; cat >/dev/null";
        let mut worker = simulated_worker(script);
        assert!(matches!(worker.infer_detailed("a", 2, &mut |_| true),
            Err(StreamInferError::Worker(message)) if message.contains("unknown protocol event")));
        assert!(worker.protocol_failed);
    }

    #[test]
    fn rejected_encoded_request_does_not_contact_or_poison_worker() {
        let mut worker = simulated_worker("cat >/dev/null");
        let prompt = "\0".repeat(60_000);
        assert!(matches!(worker.infer_detailed(&prompt, 2, &mut |_| true),
            Err(StreamInferError::Rejected(message)) if message.contains("encoded request")));
        assert!(!worker.protocol_failed);
    }
}

#[cfg(test)]
mod startup_timeout_tests {
    use super::startup_timeout_seconds;
    #[test]
    fn accepts_bounded_startup_deadline_without_infinite_blocking() {
        assert_eq!(startup_timeout_seconds(None),300);
        assert_eq!(startup_timeout_seconds(Some("45")),45);
        assert_eq!(startup_timeout_seconds(Some("0")),30);
        assert_eq!(startup_timeout_seconds(Some("99999")),900);
        assert_eq!(startup_timeout_seconds(Some("not a number")),300);
    }
}

#[cfg(test)]
mod cache_budget_tests {
    use super::native_cache_mib;
    const MIB: u64 = 1024 * 1024;

    #[test]
    fn cache_size_tracks_actual_headroom_not_total_ram() {
        assert_eq!(native_cache_mib(None), 64);
        assert_eq!(native_cache_mib(Some(0)), 0);
        assert_eq!(native_cache_mib(Some(128*MIB)), 2);
        assert_eq!(native_cache_mib(Some(512*MIB)), 8);
        assert_eq!(native_cache_mib(Some(4096*MIB)), 64);
        assert_eq!(native_cache_mib(Some(16_384*MIB)), 256);
        assert_eq!(native_cache_mib(Some(u64::MAX)), 256);
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::read_bounded_line;
    use std::io::{BufReader, Cursor};

    #[test]
    fn bounded_jsonl_reader_rejects_truncation_and_oversized_input() {
        let mut ok = BufReader::new(Cursor::new(b"{\"event\":\"ready\"}\n".to_vec()));
        assert_eq!(read_bounded_line(&mut ok, 40).unwrap().unwrap(), "{\"event\":\"ready\"}\n");
        assert!(read_bounded_line(&mut ok, 40).unwrap().is_none());

        let mut incomplete = BufReader::new(Cursor::new(b"not terminated".to_vec()));
        assert!(read_bounded_line(&mut incomplete, 64).is_err());

        let mut oversized = BufReader::new(Cursor::new(vec![b'x'; 100_000]));
        assert!(read_bounded_line(&mut oversized, 32).is_err());

        let mut bad_utf8 = BufReader::new(Cursor::new(vec![255, b'\n']));
        assert!(read_bounded_line(&mut bad_utf8, 32).is_err());
    }
}
