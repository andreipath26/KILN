//! Adapter: LlamaContext -> kiln_core::chat::Forward.
//! Phase 2.2. Lives in the CLI on purpose; kiln-runtime stays a pure
//! llama.cpp wrapper. Trait mismatch handled here: Forward::forward
//! takes &[u32] -> Vec<f32>, LlamaContext::forward takes &[i32] ->
//! Result<Vec<f32>, LlamaError>. On error we store the message in
//! last_error and return an empty Vec; the caller checks last_error.
//! No panic, no trait change. Fail-safe, Rule KILN-E7.

use kiln_core::chat::Forward;
use kiln_runtime::LlamaContext;
use std::path::Path;

pub struct LlamaForward {
    ctx: LlamaContext,
    last_error: Option<String>,
}

impl LlamaForward {
    pub fn load(model: &Path) -> Result<Self, kiln_runtime::LlamaError> {
        Ok(Self { ctx: LlamaContext::load(model)?, last_error: None })
    }
    pub fn take_error(&mut self) -> Option<String> { self.last_error.take() }
    pub fn ctx(&self) -> &LlamaContext { &self.ctx }
    pub fn ctx_mut(&mut self) -> &mut LlamaContext { &mut self.ctx }
}

impl Forward for LlamaForward {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        match self.ctx.forward(&ids) {
            Ok(l) => { self.last_error = None; l }
            Err(e) => { self.last_error = Some(e.to_string()); Vec::new() }
        }
    }
    fn vocab_size(&self) -> usize { self.ctx.n_vocab() as usize }
}
