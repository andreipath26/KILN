//! Dispatch seam. See docs/dispatch-integration-design.md.
//! Rule KILN-E36: every operation goes through the registry.

/// The quantization formats the runtime understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantKind {
    F32,
    F16,
    Q4K,
    Q6K,
    TQ1_0,
}

impl QuantKind {
    /// Bytes required to store one contiguous row of `num_cols` weights
    /// in this quant format. This is the single source of truth for
    /// row layout. No call site computes it independently.
    pub fn row_bytes(&self, num_cols: usize) -> usize {
        match self {
            QuantKind::F32 => num_cols * 4,
            QuantKind::F16 => num_cols * 2,
            QuantKind::Q4K => {
                let blocks = (num_cols + 255) / 256;
                blocks * 144
            }
            QuantKind::Q6K => {
                let blocks = (num_cols + 255) / 256;
                blocks * 210
            }
            QuantKind::TQ1_0 => (num_cols + 4) / 5,
        }
    }

    /// Detect the quant kind from a raw byte count for `num_cols`
    /// weights. Returns None if no known format matches.
    pub fn detect(num_cols: usize, byte_len: usize) -> Option<QuantKind> {
        for k in [QuantKind::TQ1_0, QuantKind::Q6K, QuantKind::Q4K, QuantKind::F16, QuantKind::F32] {
            if k.row_bytes(num_cols) == byte_len {
                return Some(k);
            }
        }
        None
    }
}

pub type TensorId = String;

#[derive(Debug, Clone)]
pub enum Operation {
    Matmul { weights: TensorId, quant: QuantKind, out_features: usize, in_features: usize },
    RmsNorm { size: usize },
    Embedding { vocab: usize, hidden: usize },
    Attention { num_heads: usize, num_kv_heads: usize, head_dim: usize, seq_len: usize },
    Sample { vocab: usize },
}

impl Operation {
    pub fn kind(&self) -> &'static str {
        match self {
            Operation::Matmul { .. } => "matmul",
            Operation::RmsNorm { .. } => "rmsnorm",
            Operation::Embedding { .. } => "embedding",
            Operation::Attention { .. } => "attention",
            Operation::Sample { .. } => "sample",
        }
    }
}

#[derive(Debug)]
pub enum DispatchError {
    NoBackend(&'static str),
    MissingTensor(String),
    MissingQuant(String),
    Backend(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::NoBackend(k) => write!(f, "no backend supports {}", k),
            DispatchError::MissingTensor(t) => write!(f, "missing tensor {}", t),
            DispatchError::MissingQuant(t) => write!(f, "missing quant kind for {}", t),
            DispatchError::Backend(m) => write!(f, "backend error: {}", m),
        }
    }
}

impl std::error::Error for DispatchError {}

pub struct Scratch {
    pub buf: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self { Self { buf: Vec::new() } }
    pub fn ensure(&mut self, n: usize) {
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
    }
}

impl Default for Scratch {
    fn default() -> Self { Self::new() }
}

pub struct ExecContext<'a> {
    /// The weight bytes for this single operation. No lookup, no clone.
    pub weight_bytes: &'a [u8],
    /// The quant kind for this weight tensor.
    pub quant: QuantKind,
    pub scratch: &'a mut Scratch,
    pub out: &'a mut [f32],
    pub activations: &'a [f32],
    pub seq_len: usize,
    pub position_offset: usize,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, op: &Operation) -> bool;
    fn estimate(&self, op: &Operation) -> Option<u64>;
    fn execute(&self, op: &Operation, ctx: &mut ExecContext) -> Result<(), DispatchError>;
}

pub struct Dispatcher {
    backends: Vec<Box<dyn Backend>>,
}

impl Dispatcher {
    pub fn new() -> Self { Self { backends: Vec::new() } }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn backend_count(&self) -> usize { self.backends.len() }

    pub fn dispatch(&self, op: &Operation, ctx: &mut ExecContext) -> Result<(), DispatchError> {
        for b in &self.backends {
            if b.supports(op) {
                return b.execute(op, ctx);
            }
        }
        Err(DispatchError::NoBackend(op.kind()))
    }
}

impl Default for Dispatcher {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBackend;

    impl Backend for DummyBackend {
        fn name(&self) -> &str { "dummy" }
        fn supports(&self, _op: &Operation) -> bool { true }
        fn estimate(&self, _op: &Operation) -> Option<u64> { Some(0) }
        fn execute(&self, _op: &Operation, _ctx: &mut ExecContext) -> Result<(), DispatchError> { Ok(()) }
    }

    fn make_ctx<'a>(
        w: &'a mut HashMap<TensorId, Vec<u8>>,
        q: &'a mut HashMap<TensorId, QuantKind>,
        s: &'a mut Scratch,
        o: &'a mut [f32],
        a: &'a [f32],
    ) -> ExecContext<'a> {
        ExecContext {
            weights: w, quants: q, scratch: s, out: o, activations: a,
            seq_len: 1, position_offset: 0,
        }
    }

    #[test]
    fn dispatcher_finds_backend() {
        let mut d = Dispatcher::new();
        d.register(Box::new(DummyBackend));
        assert_eq!(d.backend_count(), 1);
        let mut w = HashMap::new();
        let mut q = HashMap::new();
        let mut s = Scratch::new();
        let mut o = [0.0f32; 4];
        let a = [0.0f32; 4];
        let mut ctx = make_ctx(&mut w, &mut q, &mut s, &mut o, &a);
        let op = Operation::RmsNorm { size: 4 };
        assert!(d.dispatch(&op, &mut ctx).is_ok());
    }

    #[test]
    fn dispatcher_errors_without_backend() {
        let d = Dispatcher::new();
        let mut w = HashMap::new();
        let mut q = HashMap::new();
        let mut s = Scratch::new();
        let mut o = [0.0f32; 4];
        let a = [0.0f32; 4];
        let mut ctx = make_ctx(&mut w, &mut q, &mut s, &mut o, &a);
        let op = Operation::RmsNorm { size: 4 };
        assert!(matches!(d.dispatch(&op, &mut ctx), Err(DispatchError::NoBackend("rmsnorm"))));
    }
}
