//! llama.cpp runtime backend. Phase 3.
//! Wraps LlamaContext. Owns the live session.

use crate::ffi::{LlamaContext, LlamaError};
use crate::runtime_dispatch::{RuntimeBackend, RuntimeError, RuntimeOp, RuntimeResult};

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
            _ => self.ctx.is_some(),
        }
    }

    fn estimate(&self, _op: &RuntimeOp) -> Option<u64> { Some(0) }

    fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
        match op {
            RuntimeOp::Load { path, n_ctx, n_batch, n_threads } => {
                let max = i32::MAX as u32;
                if *n_ctx == 0 || *n_batch == 0 || *n_threads == 0
                    || *n_ctx > max || *n_batch > max || *n_threads > max
                {
                    return Err(RuntimeError::Llama(LlamaError::Load(
                        "n_ctx, n_batch, and n_threads must be positive values within i32 range".to_string(),
                    )));
                }
                let ctx = LlamaContext::load_with(
                    path,
                    Some((*n_ctx as i32, *n_batch as i32, *n_threads as i32)),
                )?;
                self.ctx = Some(ctx);
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::Forward { tokens } => {
                let ctx = self.ctx.as_mut().ok_or(RuntimeError::NoSession)?;
                let logits = ctx.forward(tokens)?;
                Ok(RuntimeResult::Logits(logits))
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
                Ok(RuntimeResult::Bytes(ctx.token_bytes(*token)))
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
