//! Transformer weights read from a GGUF file.
//!
//! Weights stay packed (Q4_K, Q6_K, F32). The forward pass feeds the
//! packed bytes to the fused matmul kernels.
//!
//! See docs/transformer-design.md for the design.

use std::sync::Arc;

use kiln_models::gguf::{GgufFile, GgufType};

use crate::transformer_config::{ConfigError, TransformerConfig};

#[derive(Debug)]
pub enum WeightError {
    Config(ConfigError),
    MissingTensor(String),
    WrongDtype { tensor: String, got: GgufType },
    ZeroLength,
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightError::Config(e) => write!(f, "config error: {}", e),
            WeightError::MissingTensor(n) => write!(f, "missing tensor: {}", n),
            WeightError::WrongDtype { tensor, got } => {
                write!(f, "tensor {} has wrong dtype: {:?}", tensor, got)
            }
            WeightError::ZeroLength => write!(f, "zero-length tensor"),
        }
    }
}

impl std::error::Error for WeightError {}

impl From<ConfigError> for WeightError {
    fn from(e: ConfigError) -> Self {
        WeightError::Config(e)
    }
}

/// One layer's weights. Each field is a copy of the packed bytes.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<u8>,
    pub attn_q: Vec<u8>,
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_output: Vec<u8>,
    pub ffn_norm: Vec<u8>,
    pub ffn_gate: Vec<u8>,
    pub ffn_up: Vec<u8>,
    pub ffn_down: Vec<u8>,
}

/// The full transformer weights.
#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub config: TransformerConfig,
    pub token_embd: Vec<u8>,
    pub output_norm: Vec<u8>,
    pub output: Vec<u8>,
    pub layers: Vec<LayerWeights>,
    /// Whether the output projection is tied to the token embedding.
    /// True for Qwen2.5 and most modern transformer models.
    pub tied_embeddings: bool,
}

impl TransformerWeights {
    /// Load from a GGUF file. Weights are copied into Vecs.
    pub fn from_gguf(file: &Arc<GgufFile>) -> Result<Self, WeightError> {
        let config = TransformerConfig::from_gguf(file)?;

        let get_tensor = |name: &str| -> Result<Vec<u8>, WeightError> {
            let t = file.tensor(name)
                .ok_or_else(|| WeightError::MissingTensor(name.to_string()))?;
            let bytes = file.tensor_bytes(name)
                .ok_or_else(|| WeightError::WrongDtype {
                    tensor: name.to_string(),
                    got: t.dtype,
                })?;
            Ok(bytes.to_vec())
        };

        let token_embd = get_tensor("token_embd.weight")?;
        let output_norm = get_tensor("output_norm.weight")?;

        // The output projection. Qwen2.5 and most modern transformer
        // models use tied embeddings. The GGUF omits the duplicate
        // output.weight tensor and the model is expected to reuse the
        // token embedding matrix. When output.weight is present, we use
        // it. When it is absent, the architecture requires the token
        // embedding to serve as the output projection. This is not a
        // fallback. It is what the model architecture specifies.
        let output = match file.tensor("output.weight") {
            Some(_) => get_tensor("output.weight")?,
            None => token_embd.clone(),
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{}.", i);
            let layer = LayerWeights {
                attn_norm: get_tensor(&format!("{}attn_norm.weight", p))?,
                attn_q: get_tensor(&format!("{}attn_q.weight", p))?,
                attn_k: get_tensor(&format!("{}attn_k.weight", p))?,
                attn_v: get_tensor(&format!("{}attn_v.weight", p))?,
                attn_output: get_tensor(&format!("{}attn_output.weight", p))?,
                ffn_norm: get_tensor(&format!("{}ffn_norm.weight", p))?,
                ffn_gate: get_tensor(&format!("{}ffn_gate.weight", p))?,
                ffn_up: get_tensor(&format!("{}ffn_up.weight", p))?,
                ffn_down: get_tensor(&format!("{}ffn_down.weight", p))?,
            };
            layers.push(layer);
        }

        let tied_embeddings = file.tensor("output.weight").is_none();

        Ok(Self {
            config,
            token_embd,
            output_norm,
            output,
            layers,
            tied_embeddings,
        })
    }

    pub fn total_bytes(&self) -> usize {
        let mut n = self.token_embd.len() + self.output_norm.len() + self.output.len();
        for l in &self.layers {
            n += l.attn_norm.len() + l.attn_q.len() + l.attn_k.len() + l.attn_v.len()
                + l.attn_output.len() + l.ffn_norm.len() + l.ffn_gate.len()
                + l.ffn_up.len() + l.ffn_down.len();
        }
        n
    }
}
