//! Shared inference state for the HTTP API.
//!
//! Set KILN_MODEL_PATH to a GGUF file before starting the server to enable
//! non-streaming generation and chat. Without it, inference routes fail closed.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::stream_worker::StreamWorker;
use kiln_runtime::{LlamaCppBackend, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};

pub(crate) struct RuntimeSession {
    pub dispatcher: Option<RuntimeDispatcher>,
    pub stream_worker: Option<StreamWorker>,
    pub streaming: bool,
    pub model_name: String,
    pub model_path: PathBuf,
    pub eos_token: i32,
}

#[derive(Clone)]
pub(crate) struct ApiState {
    pub runtime: Arc<Mutex<Option<RuntimeSession>>>,
    pub activation_lock: Arc<Mutex<()>>,
    pub download_semaphore: Arc<tokio::sync::Semaphore>,
}

/// Reusing a loaded model avoids expensive shutdown, SSD reads, mmap
/// setup, lost KV state and a second risk of OOM. An explicit transition from
/// resident to streaming still requires reopening through the native worker.
fn reuse_existing_session(
    current_path: &std::path::Path,
    requested_path: &std::path::Path,
    already_streaming: bool,
    force_stream: bool,
) -> bool {
    current_path == requested_path && (!force_stream || already_streaming)
}

impl ApiState {
    pub fn unconfigured() -> Self {
        Self {
            runtime: Arc::new(Mutex::new(None)),
            activation_lock: Arc::new(Mutex::new(())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    /// Load one configured model at server startup. Missing configuration is
    /// allowed so the server can expose health/version routes without inference.
    pub fn from_env() -> Result<Self, String> {
        let Some(path) = std::env::var_os("KILN_MODEL_PATH").map(PathBuf::from) else {
            return Ok(Self::unconfigured());
        };

        let metadata = std::fs::metadata(&path)
            .map_err(|e| format!("KILN_MODEL_PATH {} is not readable: {}", path.display(), e))?;
        if !metadata.is_file() {
            return Err(format!("KILN_MODEL_PATH {} is not a file", path.display()));
        }

        let path = kiln_models::library::normalize_gguf_load_path(&path)?;
        let session = load_session(path, false)?;

        Ok(Self {
            runtime: Arc::new(Mutex::new(Some(session))),
            activation_lock: Arc::new(Mutex::new(())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }

    /// Replace the active model without ever leaving a partially initialized
    /// session visible. If loading the new model fails, restore the prior one.
    pub fn activate(&self, path: PathBuf, force_stream: bool) -> Result<String, String> {
        let _activation = self.activation_lock.lock()
            .map_err(|_| "model activation lock is poisoned".to_string())?;
        // This check executes while activation is serialized, and never
        // creates a second native worker for an already-active GGUF.
        // Canonicalize paths to avoid a relative/symlink alias causing a
        // needless reload of exactly the same model.
        let normalized_path = std::fs::canonicalize(&path)
            .map_err(|e| format!("cannot activate model {}: {e}",path.display()))?;
        {
            let runtime = self.runtime.lock()
                .map_err(|_| "runtime state lock is poisoned".to_string())?;
            if let Some(active) = runtime.as_ref() {
                let current_canonical = std::fs::canonicalize(&active.model_path).ok();
                if current_canonical.as_deref().is_some_and(|current|
                    reuse_existing_session(current, &normalized_path, active.streaming, force_stream)
                ) {
                    return Ok(active.model_name.clone());
                }
            }
        }
        let previous = {
            let mut runtime = self.runtime.lock()
                .map_err(|_| "runtime state lock is poisoned".to_string())?;
            runtime.take().map(|session| (
                session.model_path.clone(), session.model_name.clone(), session.streaming
            ))
        };

        // The original named GGUF shard path must remain intact here:
        // canonicalizing a symlink to a content-addressed HF blob may erase
        // its conventional split-GGUF suffix before shard detection.
        match load_session(path, force_stream) {
            Ok(session) => {
                let name = session.model_name.clone();
                *self.runtime.lock().map_err(|_| "runtime state lock is poisoned".to_string())? = Some(session);
                Ok(name)
            }
            Err(load_error) => {
                if let Some((previous_path, previous_name, was_streaming)) = previous {
                    // Restore the original execution mode, including an
                    // explicitly streamed small MoE model. Re-selecting the
                    // ordinary loader could otherwise crash on recovery.
                    match load_session(previous_path, was_streaming) {
                        Ok(session) => {
                            *self.runtime.lock().map_err(|_| "runtime state lock is poisoned".to_string())? = Some(session);
                            Err(format!("model activation failed: {load_error}; restored previous model '{previous_name}'"))
                        }
                        Err(restore_error) => Err(format!("model activation failed: {load_error}; restoring previous model also failed: {restore_error}")),
                    }
                } else {
                    Err(format!("model activation failed: {load_error}"))
                }
            }
        }
    }
}

/// Physical free-memory and container headroom are different from installed
/// RAM. Requiring headroom avoids choosing a resident load simply because a
/// GGUF happens to fit within the machine's *total* RAM.
fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let mut fields = line.split_whitespace();
    if fields.next()? != "MemAvailable:" { return None; }
    let kib = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() { return None; }
    kib.checked_mul(1024)
}

fn finite_cgroup_bytes(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw == "max" || raw.is_empty() || !raw.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    raw.parse::<u64>().ok().filter(|n| *n > 0 && *n < (1u64 << 60))
}

fn available_ram_bytes() -> Option<u64> {
    #[cfg(target_os="linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut available = parse_mem_available_bytes(&meminfo)?;
        for (max_path, current_path) in [
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ] {
            let cap = std::fs::read_to_string(max_path).ok()
                .and_then(|s| finite_cgroup_bytes(&s));
            let usage = std::fs::read_to_string(current_path).ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            if let (Some(cap), Some(usage)) = (cap,usage) {
                available = available.min(cap.saturating_sub(usage));
            }
        }
        Some(available)
    }
    #[cfg(not(target_os="linux"))]
    { None }
}

fn resident_budget_bytes(available: u64) -> u64 {
    // Allow the OS, inference context, runtime allocations and other programs
    // to use their share. This is deliberately a conservative dispatch hint,
    // not a guarantee that the entire model would otherwise be resident.
    available.saturating_sub((available / 5).max(512 * 1024 * 1024))
}

/// Live memory estimate for the local UI; never claim the resident budget
/// equals the full amount available to the OS. Unknown values stay unknown.
pub(crate) fn memory_headroom_bytes() -> Option<(u64, u64)> {
    let available = available_ram_bytes()?;
    Some((available, resident_budget_bytes(available)))
}

fn should_stream_single_file(
    force_stream: bool, is_split: bool, model_bytes: u64, budget_bytes: u64
) -> bool {
    !is_split && (force_stream || model_bytes > budget_bytes)
}

/// A complete split-model source map is necessary, but streaming is selected
/// only when requested or when the combined GGUF exceeds available memory.
fn should_stream_split_file(
    force_stream: bool, compatible_split: bool, model_bytes: u64, budget_bytes: u64
) -> bool {
    compatible_split && (force_stream || model_bytes > budget_bytes)
}

fn load_session(path: PathBuf, force_stream: bool) -> Result<RuntimeSession, String> {
    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("model path {} is not readable: {}", path.display(), e))?;
    if !metadata.is_file() {
        return Err(format!("model path {} is not a file", path.display()));
    }

        let profile = SystemProfile::load_or_probe();
        let available = available_ram_bytes().unwrap_or_else(|| {
            u64::from(profile.ram_gb).saturating_mul(1_073_741_824)
        });
        let resident_budget = resident_budget_bytes(available);
        let is_split = kiln_models::library::split_gguf_parts(&path).is_some();
        let total_size = if let Some((prefix, _, count)) = kiln_models::library::split_gguf_parts(&path) {
            let base = prefix.file_name().and_then(|s| s.to_str()).unwrap_or("");
            (1..=count).filter_map(|n| std::fs::metadata(
                prefix.with_file_name(format!("{base}-{n:05}-of-{count:05}.gguf"))
            ).ok().map(|m| m.len())).sum::<u64>()
        } else { metadata.len() };
        if force_stream && !cfg!(target_os = "linux") {
            return Err("native CPU expert streaming is currently Linux-only".into());
        }
        let native_split_candidate = if is_split {
            let inventory = kiln_models::split_gguf::inspect_split_gguf(&path)?
                .ok_or("model has split filename but could not be indexed as split GGUF")?;
            (2..=64).contains(&inventory.shard_count) &&
                !inventory.fused_expert_slabs.is_empty() &&
                inventory.fused_expert_slabs.len() <= 300_000 &&
                inventory.unsupported_expert_tensors.is_empty()
        } else { false };
        if force_stream && is_split && !native_split_candidate {
            return Err("this split GGUF lacks a complete supported fused MoE expert layout for native multi-source streaming".into());
        }
        if cfg!(target_os = "linux") && (
            should_stream_single_file(force_stream, is_split, total_size, resident_budget) ||
            should_stream_split_file(force_stream, native_split_candidate, total_size, resident_budget)
        ) {
            let worker = StreamWorker::start(&path).map_err(|e| format!(
                "model ({:.1} GiB; resident budget {:.1} GiB) was routed to native expert streaming, but the GGUF layout or streaming worker is incompatible: {e}",
                total_size as f64 / 1_073_741_824.0,
                resident_budget as f64 / 1_073_741_824.0
            ))?;
            let model_name = path.file_stem().and_then(|s| s.to_str())
                .unwrap_or("model").to_owned();
            return Ok(RuntimeSession {
                dispatcher: None, stream_worker: Some(worker), streaming: true,
                model_name, model_path: path, eos_token: -1,
            });
        }
        let mut dispatcher = RuntimeDispatcher::new();
        dispatcher.register(Box::new(LlamaCppBackend::new()));
        dispatcher.dispatch(RuntimeOp::Load {
            path: path.clone(),
            n_ctx: profile.n_ctx,
            n_batch: profile.n_batch,
            n_ubatch: profile.n_batch.min(512),
            n_threads: profile.n_threads,
            lazy_moe_experts: total_size > resident_budget,
        }).map_err(|e| format!("failed to load configured model {}: {}", path.display(), e))?;

        let eos_token = match dispatcher.dispatch(RuntimeOp::EosToken)
            .map_err(|e| format!("failed to read model EOS token: {}", e))? {
            RuntimeResult::Int(token) => token,
            _ => return Err("runtime returned an unexpected EOS-token result".to_string()),
        };
        let model_name = path.file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("default")
            .to_string();

        tracing::info!(
            model = %model_name,
            path = %path.display(),
            n_ctx = profile.n_ctx,
            n_batch = profile.n_batch,
            n_threads = profile.n_threads,
            "loaded API inference model"
        );

        Ok(RuntimeSession {
                dispatcher: Some(dispatcher),
                stream_worker: None,
                streaming: false,
                model_name,
                model_path: path,
                eos_token,
            })
}

#[cfg(test)]
mod session_reuse_tests {
    use super::reuse_existing_session;
    use std::path::Path;
    #[test]
    fn same_model_does_not_restart_the_streamed_worker() {
        let model=Path::new("/models/big-00001-of-00002.gguf");
        assert!(reuse_existing_session(model, model, true, false));
        assert!(reuse_existing_session(model, model, true, true));
        assert!(!reuse_existing_session(model, model, false, true));
        assert!(reuse_existing_session(model, model, false, false));
        assert!(!reuse_existing_session(model, Path::new("/models/other.gguf"), true, false));
    }
}

#[cfg(test)]
mod memory_routing_tests {
    use super::*;

    #[test]
    fn reads_memavailable_without_confusing_total_ram() {
        let sample = "MemTotal:       16384000 kB\nMemFree: 1024 kB\nMemAvailable: 2000000 kB\n";
        assert_eq!(parse_mem_available_bytes(sample), Some(2_048_000_000));
        assert_eq!(parse_mem_available_bytes("MemAvailable: max kB"), None);
        assert_eq!(parse_mem_available_bytes("MemAvailable: 123 bytes"), None);
        assert_eq!(parse_mem_available_bytes("MemAvailable: 0 kB"), Some(0));
        assert_eq!(parse_mem_available_bytes("MemAvailable: 18446744073709551615 kB"), None);
        // A genuinely exhausted host must NEVER fall back to installed RAM.
        assert_eq!(resident_budget_bytes(parse_mem_available_bytes("MemAvailable: 0 kB").unwrap()), 0);
    }

    #[test]
    fn parses_only_real_cgroup_limits() {
        assert_eq!(finite_cgroup_bytes("max"),None);
        assert_eq!(finite_cgroup_bytes("9223372036854771712"),None);
        assert_eq!(finite_cgroup_bytes("2147483648\n"),Some(2_147_483_648));
        assert_eq!(finite_cgroup_bytes("-1"),None);
    }

    #[test]
    fn conservative_budget_reserves_memory_for_os_and_kv() {
        let gib = 1_073_741_824u64;
        assert_eq!(resident_budget_bytes(16*gib), 16*gib - 16*gib/5);
        assert_eq!(resident_budget_bytes(gib), gib - 512*1024*1024);
        assert_eq!(resident_budget_bytes(128*1024*1024), 0);
        assert_eq!(resident_budget_bytes(0), 0);
    }

    #[test]
    fn compatible_split_moe_automatically_selects_native_streaming_under_pressure() {
        let gib = 1_073_741_824u64;
        // The *entire* source GGUF set is compared with the RAM budget.
        assert!(should_stream_split_file(false,true,28*gib,12*gib));
        assert!(!should_stream_split_file(false,true,6*gib,12*gib));
        // A user may opt into native expert reads for a smaller supported MoE.
        assert!(should_stream_split_file(true,true,6*gib,12*gib));
        // A missing expert projection or unsupported quantizer must not be
        // interpreted as a verified native multi-source layout.
        assert!(!should_stream_split_file(true,false,28*gib,12*gib));
        assert!(!should_stream_split_file(false,false,28*gib,12*gib));
        assert!(should_stream_split_file(false,true,28*gib,0));
    }

    #[test]
    fn available_memory_routes_oversized_single_file_not_just_total_ram() {
        let gib = 1_073_741_824u64;
        let free_budget = resident_budget_bytes(8*gib);
        // This 10 GiB model is below a 16 GiB *total* RAM figure, but
        // cannot reasonably be loaded with only 8 GiB currently available.
        assert!(should_stream_single_file(false,false,10*gib,free_budget));
        assert!(!should_stream_single_file(false,false,2*gib,free_budget));
        assert!(should_stream_single_file(true,false,2*gib,free_budget));
        // The single-file path excludes split sets; verified split models
        // are routed by should_stream_split_file instead.
        assert!(!should_stream_single_file(true,true,10*gib,free_budget));
        assert!(!should_stream_single_file(false,true,10*gib,free_budget));
    }
}
