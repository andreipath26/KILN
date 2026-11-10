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
        let previous = {
            let mut runtime = self.runtime.lock()
                .map_err(|_| "runtime state lock is poisoned".to_string())?;
            runtime.take().map(|session| (session.model_path.clone(), session.model_name.clone()))
        };

        match load_session(path.clone(), force_stream) {
            Ok(session) => {
                let name = session.model_name.clone();
                *self.runtime.lock().map_err(|_| "runtime state lock is poisoned".to_string())? = Some(session);
                Ok(name)
            }
            Err(load_error) => {
                if let Some((previous_path, previous_name)) = previous {
                    match load_session(previous_path, false) {
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

fn load_session(path: PathBuf, force_stream: bool) -> Result<RuntimeSession, String> {
    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("model path {} is not readable: {}", path.display(), e))?;
    if !metadata.is_file() {
        return Err(format!("model path {} is not a file", path.display()));
    }

        let profile = SystemProfile::load_or_probe();
        let stream_threshold = u64::from(profile.ram_gb).saturating_mul(1_073_741_824);
        let is_split = kiln_models::library::split_gguf_parts(&path).is_some();
        let total_size = if let Some((prefix, _, count)) = kiln_models::library::split_gguf_parts(&path) {
            let base = prefix.file_name().and_then(|s| s.to_str()).unwrap_or("");
            (1..=count).filter_map(|n| std::fs::metadata(
                prefix.with_file_name(format!("{base}-{n:05}-of-{count:05}.gguf"))
            ).ok().map(|m| m.len())).sum::<u64>()
        } else { metadata.len() };
        if force_stream && is_split {
            return Err("native single-file MoE streaming does not yet support split GGUFs; use the automatic lazy mapped loader when all shards are present".into());
        }
        if force_stream || (stream_threshold > 0 && total_size > stream_threshold && !is_split) {
            let worker = StreamWorker::start(&path).map_err(|e| format!(
                "oversized model ({:.1} GiB) needs native expert streaming, but it cannot be loaded in this experimental mode: {e}",
                total_size as f64 / 1_073_741_824.0
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
            lazy_moe_experts: total_size > stream_threshold && stream_threshold > 0,
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
