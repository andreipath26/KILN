//! Runtime dispatch. Phase 3. See docs/runtime-dispatch-design.md.
//! Rule KILN-E36 applies at the runtime layer, not only the kernel
//! layer. Every model operation goes through this dispatcher.

use std::path::PathBuf;

use crate::ffi::LlamaError;

#[derive(Debug, Clone)]
pub enum RuntimeOp {
    Load { path: PathBuf, n_ctx: u32, n_batch: u32, n_threads: u32 },
    Forward { tokens: Vec<i32> },
    Reset,
    Tokenize { text: String },
    Logits,
    TokenToStr { token: i32 },
    Unload,
    NVocab,
    EosToken,
}

#[derive(Debug)]
pub enum RuntimeResult {
    Unit,
    Logits(Vec<f32>),
    Tokens(Vec<i32>),
    Str(String),
    Int(i32),
}

#[derive(Debug)]
pub enum RuntimeError {
    NoBackend,
    NoSession,
    Llama(LlamaError),
    WrongReturn,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::NoBackend => write!(f, "no backend supports this operation"),
            RuntimeError::NoSession => write!(f, "no live session; call Load first"),
            RuntimeError::Llama(e) => write!(f, "{}", e),
            RuntimeError::WrongReturn => write!(f, "backend returned an unexpected result"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<LlamaError> for RuntimeError {
    fn from(e: LlamaError) -> Self { RuntimeError::Llama(e) }
}

pub trait RuntimeBackend: Send {
    fn name(&self) -> &str;
    fn supports(&self, op: &RuntimeOp) -> bool;
    fn estimate(&self, op: &RuntimeOp) -> Option<u64>;
    fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError>;
}

pub struct RuntimeDispatcher {
    backends: Vec<Box<dyn RuntimeBackend>>,
}

impl RuntimeDispatcher {
    pub fn new() -> Self { Self { backends: Vec::new() } }

    pub fn register(&mut self, backend: Box<dyn RuntimeBackend>) {
        self.backends.push(backend);
    }

    pub fn backend_count(&self) -> usize { self.backends.len() }

    pub fn dispatch(&mut self, op: RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
        for b in self.backends.iter_mut() {
            if b.supports(&op) {
                return b.execute(&op);
            }
        }
        Err(RuntimeError::NoBackend)
    }
}

impl Default for RuntimeDispatcher {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBackend { live: bool }

    impl RuntimeBackend for DummyBackend {
        fn name(&self) -> &str { "dummy" }
        fn supports(&self, op: &RuntimeOp) -> bool {
            match op {
                RuntimeOp::Load { .. } => true,
                _ => self.live,
            }
        }
        fn estimate(&self, _op: &RuntimeOp) -> Option<u64> { Some(0) }
        fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
            match op {
                RuntimeOp::Load { .. } => { self.live = true; Ok(RuntimeResult::Unit) }
                RuntimeOp::Reset => Ok(RuntimeResult::Unit),
                _ => Err(RuntimeError::NoSession),
            }
        }
    }

    #[test]
    fn dispatcher_finds_backend() {
        let mut d = RuntimeDispatcher::new();
        d.register(Box::new(DummyBackend { live: false }));
        assert_eq!(d.backend_count(), 1);
        let op = RuntimeOp::Load {
            path: PathBuf::from("/dev/null"),
            n_ctx: 4096, n_batch: 512, n_threads: 4,
        };
        assert!(d.dispatch(op).is_ok());
    }

    #[test]
    fn dispatcher_errors_without_backend() {
        let mut d = RuntimeDispatcher::new();
        let op = RuntimeOp::NVocab;
        assert!(matches!(d.dispatch(op), Err(RuntimeError::NoBackend)));
    }
}
