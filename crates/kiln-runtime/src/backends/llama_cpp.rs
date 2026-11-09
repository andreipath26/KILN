//! llama.cpp runtime backend. Phase 3.
//! Wraps LlamaContext. Owns the live session.

use crate::ffi::{LlamaContext, LlamaError};
use crate::runtime_dispatch::{RuntimeBackend, RuntimeError, RuntimeOp, RuntimeResult};
use std::sync::Arc;

pub struct LlamaCppBackend {
    ctx: Option<LlamaContext>,
}

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self { ctx: None }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self { Self::new() }
}

impl RuntimeBackend for LlamaCppBackend {
    fn name(&self) -> &str { "llama_cpp" }

    fn supports(&self, op: &RuntimeOp) -> bool {
        match op {
            RuntimeOp::Load { .. } => true,
            RuntimeOp::MoEWarmup { .. } | RuntimeOp::MoEWarmupHistory { .. }
                | RuntimeOp::MoETracePredict { .. } | RuntimeOp::MoEBeginToken { .. }
                | RuntimeOp::MoEPrefetch { .. } | RuntimeOp::MoEPoll
                | RuntimeOp::MoECommit { .. } => false,
            _ => self.ctx.is_some(),
        }
    }

    fn estimate(&self, _op: &RuntimeOp) -> Option<u64> { Some(0) }

    fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
        match op {
            RuntimeOp::MoEWarmup { .. } | RuntimeOp::MoEWarmupHistory { .. }
                | RuntimeOp::MoETracePredict { .. } | RuntimeOp::MoEBeginToken { .. }
                | RuntimeOp::MoEPrefetch { .. } | RuntimeOp::MoEPoll
                | RuntimeOp::MoECommit { .. } => Err(RuntimeError::NoBackend),
            RuntimeOp::Load { path, n_ctx, n_batch, n_ubatch, n_threads, lazy_moe_experts } => {
                let max = i32::MAX as u32;
                if *n_ctx == 0 || *n_batch == 0 || *n_ubatch == 0 || *n_threads == 0
                    || *n_ctx > max || *n_batch > max || *n_ubatch > max || *n_threads > max
                    || *n_ubatch > *n_batch
                {
                    return Err(RuntimeError::Llama(LlamaError::Load(
                        "n_ctx, n_batch, n_ubatch, and n_threads must be valid positive values within i32 range, and n_ubatch <= n_batch".to_string(),
                    )));
                }
                let ctx = LlamaContext::load_with_ubatch_and_lazy_moe(
                    path,
                    Some((*n_ctx as i32, *n_batch as i32, *n_ubatch as i32, *n_threads as i32)),
                    *lazy_moe_experts,
                )?;
                self.ctx = Some(ctx);
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::Forward { tokens } => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                let logits = ctx.forward(tokens)?;
                Ok(RuntimeResult::Logits(logits))
            }
            RuntimeOp::ForwardGreedy { tokens } => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::Int(ctx.forward_greedy(tokens)?))
            }
            RuntimeOp::VerifyGreedyDraft { tokens } => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                let (accepted, next) = ctx.verify_greedy_draft(tokens)?;
                Ok(RuntimeResult::VerifiedGreedy { accepted, next })
            }
            RuntimeOp::SaveKv { max_bytes } => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::KvSnapshot(Arc::new(ctx.snapshot_kv(*max_bytes)?)))
            }
            RuntimeOp::RestoreKv { snapshot } => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                ctx.restore_kv(snapshot)?;
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::Reset => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                ctx.reset()?;
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::Tokenize { text } => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                let ids = ctx.tokenize(text)?;
                Ok(RuntimeResult::Tokens(ids))
            }
            RuntimeOp::Logits => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::Logits(ctx.logits()?))
            }
            RuntimeOp::TokenToStr { token } => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::Bytes(ctx.token_bytes(*token)?))
            }
            RuntimeOp::Unload => {
                self.ctx = None;
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::NVocab => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::Int(ctx.n_vocab()))
            }
            RuntimeOp::EosToken => {
                let ctx = self.ctx.as_ref().ok_or(RuntimeError::NoSession)?;
                Ok(RuntimeResult::Int(ctx.eos_token()))
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn streamed_operations_are_never_claimed_or_executed_by_resident_backend() {
        let mut backend = LlamaCppBackend::new();
        let operations = [
            RuntimeOp::MoEWarmup { experts: vec![], max_bytes: 16 },
            RuntimeOp::MoEWarmupHistory {
                path: PathBuf::from("not-a-real-history.json"),
                actual_model_sha256: "a".repeat(64),
            },
            RuntimeOp::MoETracePredict {
                epoch: 1, current: kiln_mem::ExpertKey { layer: 0, expert_id: 0 },
                candidates: vec![], top_k: 0,
            },
            RuntimeOp::MoEBeginToken { epoch: 1 },
            RuntimeOp::MoEPrefetch { epoch: 1, experts: vec![] },
            RuntimeOp::MoEPoll,
            RuntimeOp::MoECommit { epoch: 1, actual: vec![] },
        ];
        for op in operations {
            assert!(!backend.supports(&op));
            assert!(matches!(backend.execute(&op), Err(RuntimeError::NoBackend)));
        }
    }

    #[test]
    fn kv_state_capture_requires_a_live_model_context() {
        let mut backend = LlamaCppBackend::new();
        let operation = RuntimeOp::SaveKv { max_bytes: 1024 };
        assert!(!backend.supports(&operation));
        assert!(matches!(backend.execute(&operation), Err(RuntimeError::NoSession)));
    }

    #[test]
    fn rejects_zero_parameters_before_native_load() {
        let mut backend = LlamaCppBackend::new();
        let result = backend.execute(&RuntimeOp::Load {
            path: PathBuf::from("not-used.gguf"),
            n_ctx: 0,
            n_batch: 512,
            n_ubatch: 256,
            n_threads: 4,
            lazy_moe_experts: false,
        });
        assert!(matches!(result, Err(RuntimeError::Llama(LlamaError::Load(_)))));
    }

    #[test]
    fn rejects_microbatch_larger_than_logical_batch() {
        let mut backend = LlamaCppBackend::new();
        let result = backend.execute(&RuntimeOp::Load {
            path: PathBuf::from("not-used.gguf"),
            n_ctx: 4096,
            n_batch: 256,
            n_ubatch: 512,
            n_threads: 4,
            lazy_moe_experts: false,
        });
        assert!(matches!(result, Err(RuntimeError::Llama(LlamaError::Load(_)))));
    }

    #[test]
    fn rejects_parameters_outside_c_int_range_before_native_load() {
        let mut backend = LlamaCppBackend::new();
        let result = backend.execute(&RuntimeOp::Load {
            path: PathBuf::from("not-used.gguf"),
            n_ctx: i32::MAX as u32 + 1,
            n_batch: 512,
            n_ubatch: 256,
            n_threads: 4,
            lazy_moe_experts: false,
        });
        assert!(matches!(result, Err(RuntimeError::Llama(LlamaError::Load(_)))));
    }
}
