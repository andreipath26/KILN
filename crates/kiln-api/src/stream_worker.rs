//! Isolated native expert streaming worker used only by the loopback UI.
//! Runs the verified `kiln stream` launcher instead of mutating process-wide
//! streaming environment variables inside the API server.
use std::io::{BufRead, BufReader, Write};
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
        let mut child = Command::new(exe)
            .arg("stream").arg(path)
            .arg("--stdio-json").arg("--parallel-reads")
            .arg("--cache-mib").arg("64")
            .stdin(Stdio::piped()).stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("could not start native streaming worker: {e}"))?;
        let input = child.stdin.take().ok_or("stream worker stdin unavailable")?;
        let output = child.stdout.take().ok_or("stream worker stdout unavailable")?;
        let mut worker = Self { child, input: Some(input), output: BufReader::new(output) };
        let mut line = String::new();
        let count = worker.output.read_line(&mut line)
            .map_err(|e| format!("stream worker handshake error: {e}"))?;
        if count == 0 || count > 4096 {
            return Err("experimental native streaming could not initialize this model; check KILN server logs for the exact unsupported GGUF or resource error".into());
        }
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
        loop {
            let mut line = String::new();
            let count = self.output.read_line(&mut line)
                .map_err(|e| format!("streamed model process read error: {e}"))?;
            if count == 0 {
                return Err("native streaming worker exited during generation; see server logs".into());
            }
            if count > 128 * 1024 { return Err("stream worker returned an oversized response".into()); }
            let response: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("malformed streamed response: {e}"))?;
            match response.get("event").and_then(|x| x.as_str()) {
                Some("chunk") => {
                    let piece = response.get("text").and_then(|x| x.as_str())
                        .ok_or("stream worker emitted invalid text")?;
                    combined.push_str(piece);
                    if combined.len() > 4 * 1024 * 1024 {
                        return Err("stream worker response exceeds safe limit".into());
                    }
                    let _ = on_chunk(piece);
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
        if self.child.try_wait().ok().flatten().is_none() {
            // The wrapper may still be exiting after EOF. It is a child of
            // this API process; do not leave it alive indefinitely.
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}
