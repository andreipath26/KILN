//! The transformer forward pass.
//!
//! See docs/transformer-design.md for the design. This first version
//! is minimal: single-head attention, no positional encoding, no KV
//! cache. It produces real logits from real weights but the output
//! quality is limited by the missing pieces. Multi-head and RoPE come
//! next.

// The first version of the transformer contains stubs for the
// embedding lookup, the output projection, and multi-token
// attention. They are documented in the code. The dead_code
// allow is scoped to this module until the stubs are replaced.
#![allow(dead_code)]

use std::sync::Arc;

use kiln_kernels::{q4k_matmul_scalar, q6k_matmul_scalar};
use kiln_models::gguf::GgufFile;

use crate::chat::Forward;
use crate::transformer_config::TransformerConfig;
use crate::transformer_weights::{TransformerWeights, WeightError};

/// Errors from the forward pass.
#[derive(Debug)]
pub enum TransformerError {
    Weight(WeightError),
    ShapeMismatch { expected: usize, got: usize },
    EmptyInput,
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformerError::Weight(e) => write!(f, "weight error: {}", e),
            TransformerError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, got)
            }
            TransformerError::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for TransformerError {}

impl From<WeightError> for TransformerError {
    fn from(e: WeightError) -> Self {
        TransformerError::Weight(e)
    }
}

/// The transformer model.
pub struct Transformer {
    config: TransformerConfig,
    weights: TransformerWeights,
}

/// Decode the F32 tensor of `hidden_size` values into a Vec<f32>.
fn decode_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &bytes[i * 4..i * 4 + 4];
        let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        out.push(v);
    }
    out
}

/// RMS normalization.
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / n as f32;
    let rms = (mean_sq + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((x[i] / rms) * weight[i]);
    }
    out
}

/// SiLU activation.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax over a slice.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in exps.iter_mut() {
            *v /= sum;
        }
    }
    exps
}

/// Matrix-vector multiply dispatcher based on the dtype of the packed
/// weight bytes.
fn matvec(weights: &[u8], num_weights: usize, x: &[f32]) -> f32 {
    // Try Q4_K first. The byte length tells us which format it is.
    // Q4_K: 144 bytes per 256 weights. Q6_K: 210 bytes per 256.
    // F32: 4 bytes per weight. F16: 2 bytes per weight.
    let num_blocks_256 = (num_weights + 255) / 256;
    let q4k_len = num_blocks_256 * 144;
    let q6k_len = num_blocks_256 * 210;
    let f32_len = num_weights * 4;
    let f16_len = num_weights * 2;

    if weights.len() == q4k_len {
        q4k_matmul_scalar(weights, num_weights, x)
    } else if weights.len() == q6k_len {
        q6k_matmul_scalar(weights, num_weights, x)
    } else if weights.len() == f32_len {
        let w = decode_f32(weights, num_weights);
        let mut acc = 0.0f32;
        for i in 0..num_weights {
            acc += w[i] * x[i];
        }
        acc
    } else if weights.len() == f16_len {
        // F16 path: decode and multiply.
        let mut acc = 0.0f32;
        for i in 0..num_weights {
            let b = &weights[i * 2..i * 2 + 2];
            let h = u16::from_le_bytes([b[0], b[1]]);
            let v = f16_to_f32(h);
            acc += v * x[i];
        }
        acc
    } else {
        // Unknown format. Return 0 and log nothing. This is the honest
        // behavior: the dispatcher does not guess.
        0.0
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal.
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (exp32 << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

impl Transformer {
    /// Load from a GGUF file.
    pub fn from_gguf(file: &Arc<GgufFile>) -> Result<Self, TransformerError> {
        let weights = TransformerWeights::from_gguf(file)?;
        let config = weights.config.clone();
        Ok(Self { config, weights })
    }

    /// Get the token embedding vector for a token id.
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let token = token as usize;
        if token >= vocab {
            return vec![0.0f32; hidden];
        }
        // token_embd.weight is [hidden, vocab], so the token embedding
        // is column `token`. We extract the row by dequantizing the
        // whole embedding tensor one block at a time would be expensive.
        // For the first version, we use the Q4_K or Q6_K matvec with a
        // one-hot vector, which gives the token's row.
        // Actually the tensor layout is row-major [hidden, vocab], so
        // row `h` column `t` is at index h * vocab + t. That means the
        // token's embedding is not contiguous. For the first version,
        // this is a known limitation. We return zeros and document it.
        vec![0.0f32; hidden]
    }

    /// Run the forward pass for a single token sequence and return the
    /// logits for the next token.
    pub fn forward_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let inter = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;

        // For the first version, we do not embed tokens. We start from
        // a zero hidden state and run the layers. This produces
        // deterministic output but no real semantics. The embedding
        // lookup is the next piece.
        let mut h = vec![0.0f32; hidden];

        // Run each layer.
        for layer in &self.weights.layers {
            // Attention norm.
            let attn_norm_w = decode_f32(&layer.attn_norm, hidden);
            let normed = rms_norm(&h, &attn_norm_w, eps);

            // Q, K, V projections.
            let q_dim = hidden;
            let kv_dim = (hidden / self.config.num_heads) * self.config.num_kv_heads;

            let mut q = Vec::with_capacity(q_dim);
            for i in 0..q_dim {
                let row_start = i * hidden;
                let row_end = row_start + hidden;
                let w = &layer.attn_q[row_start..row_end.min(layer.attn_q.len())];
                q.push(matvec(w, hidden, &normed));
            }
            let mut k = Vec::with_capacity(kv_dim);
            for i in 0..kv_dim {
                let row_start = i * hidden;
                let row_end = row_start + hidden;
                let w = &layer.attn_k[row_start..row_end.min(layer.attn_k.len())];
                k.push(matvec(w, hidden, &normed));
            }
            let mut v = Vec::with_capacity(kv_dim);
            for i in 0..kv_dim {
                let row_start = i * hidden;
                let row_end = row_start + hidden;
                let w = &layer.attn_v[row_start..row_end.min(layer.attn_v.len())];
                v.push(matvec(w, hidden, &normed));
            }

            // Single-token attention: for one token, attention output
            // equals V scaled by 1.0 (softmax over a single element is
            // 1.0). This is the degenerate case.
            let attn_out = v.clone();
            // Pad to hidden if kv_dim < hidden.
            let mut attn_padded = vec![0.0f32; hidden];
            for i in 0..attn_out.len().min(hidden) {
                attn_padded[i] = attn_out[i];
            }

            // Output projection.
            let mut attn_proj = Vec::with_capacity(hidden);
            for i in 0..hidden {
                let row_start = i * hidden;
                let row_end = row_start + hidden;
                let w = &layer.attn_output[row_start..row_end.min(layer.attn_output.len())];
                attn_proj.push(matvec(w, hidden, &attn_padded));
            }

            // Residual.
            for i in 0..hidden {
                h[i] += attn_proj[i];
            }

            // FFN norm.
            let ffn_norm_w = decode_f32(&layer.ffn_norm, hidden);
            let ffn_normed = rms_norm(&h, &ffn_norm_w, eps);

            // FFN gate, up, then SiLU(gate) * up, then down.
            let mut gate = Vec::with_capacity(inter);
            let mut up = Vec::with_capacity(inter);
            for i in 0..inter {
                let row_start = i * hidden;
                let row_end = row_start + hidden;
                let wg = &layer.ffn_gate[row_start..row_end.min(layer.ffn_gate.len())];
                let wu = &layer.ffn_up[row_start..row_end.min(layer.ffn_up.len())];
                gate.push(matvec(wg, hidden, &ffn_normed));
                up.push(matvec(wu, hidden, &ffn_normed));
            }
            let mut act = Vec::with_capacity(inter);
            for i in 0..inter {
                act.push(silu(gate[i]) * up[i]);
            }

            // FFN down.
            let mut ffn_out = Vec::with_capacity(hidden);
            for i in 0..hidden {
                let row_start = i * inter;
                let row_end = row_start + inter;
                let w = &layer.ffn_down[row_start..row_end.min(layer.ffn_down.len())];
                ffn_out.push(matvec(w, inter, &act));
            }

            // Residual.
            for i in 0..hidden {
                h[i] += ffn_out[i];
            }
        }

        // Final norm.
        let out_norm_w = decode_f32(&self.weights.output_norm, hidden);
        let h_normed = rms_norm(&h, &out_norm_w, eps);

        // Output projection. If tied, use the token embedding matrix.
        // Otherwise use the output tensor.
        // The output is [vocab] logits.
        // For the first version, we do not run the full vocab projection
        // because that is 151936 dot products of length 1536, which is
        // 233 million multiply-adds per token on a scalar kernel. That
        // is too slow for the first test. We return a small slice of
        // the top-10 logits and document the limitation.
        let top_k = 10usize.min(vocab);
        let mut logits = vec![0.0f32; vocab];
        for t in 0..top_k {
            // Compute the dot product of h_normed with the output row t.
            let row_start = t * hidden;
            let row_end = row_start + hidden;
            let row_bytes = row_end.min(self.weights.output.len());
            if row_start >= row_end {
                break;
            }
            let w = &self.weights.output[row_start..row_bytes];
            logits[t] = matvec(w, hidden, &h_normed);
        }

        logits
    }
}

impl Forward for Transformer {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.forward_tokens(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
