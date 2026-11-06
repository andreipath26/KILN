//! Runtime dispatch. Phase 3. See docs/runtime-dispatch-design.md.
//! Rule KILN-E36 applies at the runtime layer, not only the kernel
//! layer. Every model operation goes through this dispatcher.

use std::path::PathBuf;
use std::sync::Arc;
use kiln_mem::ExpertKey;

use crate::ffi::LlamaError;

#[derive(Debug, Clone)]
pub enum RuntimeOp {
    Load { path: PathBuf, n_ctx: u32, n_batch: u32, n_ubatch: u32, n_threads: u32 },
    Forward { tokens: Vec<i32> },
    /// Greedy-only path. Returns RuntimeResult::Int(token_id), not logits.
    ForwardGreedy { tokens: Vec<i32> },
    /// Batched greedy draft verification, with rejected KV tail rolled back.
    VerifyGreedyDraft { tokens: Vec<i32> },
    /// Advance the token-scoped expert Commit Guard.
    MoEBeginToken { epoch: u64 },
    /// Schedule optional predicted expert reads; these are not authoritative.
    MoEPrefetch { epoch: u64, experts: Vec<ExpertKey> },
    /// Poll completed speculative I/O without blocking the decode thread.
    MoEPoll,
    /// Only the actual router keys can authorize bytes for compute.
    MoECommit { epoch: u64, actual: Vec<ExpertKey> },
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
    /// Accepted prefix length and the unevaluated corrected next token.
    VerifiedGreedy { accepted: usize, next: i32 },
    MoEPrefetchSubmitted { submitted: usize, rejected: usize },
    MoEStaged { count: usize },
    MoEExpertBytes {
        experts: Vec<(ExpertKey, Arc<[u8]>)>,
        predicted_hits: usize,
        authoritative_reads: usize,
    },
    Str(String),
    Bytes(Vec<u8>),
    Int(i32),
}

#[derive(Debug)]
pub enum RuntimeError {
    NoBackend,
    NoSession,
    Llama(LlamaError),
    WrongReturn,
    ExpertStream(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::NoBackend => write!(f, "no backend supports this operation"),
            RuntimeError::NoSession => write!(f, "no live session; call Load first"),
            RuntimeError::Llama(e) => write!(f, "{}", e),
            RuntimeError::WrongReturn => write!(f, "backend returned an unexpected result"),
            RuntimeError::ExpertStream(reason) => write!(f, "streamed expert backend: {reason}"),
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
            n_ctx: 4096, n_batch: 512, n_ubatch: 256, n_threads: 4,
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
