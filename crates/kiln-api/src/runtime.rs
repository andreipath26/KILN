//! Shared inference state for the HTTP API.
//!
//! Set KILN_MODEL_PATH to a GGUF file before starting the server to enable
//! non-streaming generation and chat. Without it, inference routes fail closed.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use kiln_runtime::{LlamaCppBackend, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};

pub(crate) struct RuntimeSession {
    pub dispatcher: RuntimeDispatcher,
    pub model_name: String,
    pub model_path: PathBuf,
    pub eos_token: i32,
}

#[derive(Clone)]
pub(crate) struct ApiState {
    pub runtime: Arc<Mutex<Option<RuntimeSession>>>,
}

impl ApiState {
    pub fn unconfigured() -> Self {
        Self {
            runtime: Arc::new(Mutex::new(None)),
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

        let profile = SystemProfile::load_or_probe();
        let mut dispatcher = RuntimeDispatcher::new();
        dispatcher.register(Box::new(LlamaCppBackend::new()));
        dispatcher.dispatch(RuntimeOp::Load {
            path: path.clone(),
            n_ctx: profile.n_ctx,
            n_batch: profile.n_batch,
            n_threads: profile.n_threads,
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

        Ok(Self {
            runtime: Arc::new(Mutex::new(Some(RuntimeSession {
                dispatcher,
                model_name,
                model_path: path,
                eos_token,
            }))),
        })
    }
}
