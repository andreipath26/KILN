//! Isolated native expert streaming worker used only by the loopback UI.
//! Runs the verified `kiln stream` launcher instead of mutating process-wide
//! streaming environment variables inside the API server.
use std::io::{BufRead, BufReader, Read, Write};
use std::time::{Duration, Instant};

const MAX_IPC_LINE: usize = 128 * 1024;
const MAX_HANDSHAKE_LINE: usize = 4096;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

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

pub struct StreamWorker {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
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
        let mut child = command
            .arg("stream").arg(path)
            .arg("--stdio-json").arg("--parallel-reads")
            .arg("--cache-mib").arg("64")
            .stdin(Stdio::piped()).stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("could not start native streaming worker: {e}"))?;
        let input = child.stdin.take().ok_or("stream worker stdin unavailable")?;
        let output = child.stdout.take().ok_or("stream worker stdout unavailable")?;
        let mut worker = Self { child, input: Some(input), output: BufReader::new(output) };
        let line = read_bounded_line(&mut worker.output, MAX_HANDSHAKE_LINE)?
            .ok_or("experimental native streaming could not initialize this model; check KILN server logs for its GGUF or resource error")?;
        let response: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("malformed stream worker handshake: {e}"))?;
        if response.get("event").and_then(|s| s.as_str()) != Some("ready") {
            return Err("native streaming worker did not confirm model loading".into());
        }
        Ok(worker)
    }

    pub fn infer(&mut self, prompt: &str, max_tokens: usize,
                 on_chunk: &mut dyn FnMut(&str) -> bool) -> Result<String, String> {
        if prompt.is_empty() || prompt.len() > 131_072 || max_tokens == 0 || max_tokens > 4096 {
            return Err("streamed prompt or token count exceeds bounded IPC limits".into());
        }
        let request = serde_json::json!({"prompt":prompt,"max_tokens":max_tokens});
        let input = self.input.as_mut().ok_or("stream worker stdin has closed")?;
        writeln!(input,"{request}").and_then(|_| input.flush())
            .map_err(|e| format!("streamed model process is unavailable: {e}"))?;
        let mut combined = String::new();
        let mut receiver_open = true;
        loop {
            let line = read_bounded_line(&mut self.output, MAX_IPC_LINE)?
                .ok_or("native streaming worker exited during generation; see server logs")?;
            let response: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("malformed streamed response: {e}"))?;
            match response.get("event").and_then(|x| x.as_str()) {
                Some("chunk") => {
                    let piece = response.get("text").and_then(|x| x.as_str())
                        .ok_or("stream worker emitted invalid text")?;
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
                Some("done") => break,
                _ => return Err("stream worker emitted an unknown protocol event".into()),
            }
        }
        Ok(combined)
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
                    #[cfg(unix)]
                    {
                        // Kill the entire dedicated process group, including
                        // the nested native compute worker.
                        unsafe extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
                        if let Ok(pid) = i32::try_from(self.child.id()) {
                            unsafe { let _ = kill(-pid, 9); }
                        }
                    }
                    let _ = self.child.kill();
                    break;
                }
            }
        }
        let _ = self.child.wait();
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
