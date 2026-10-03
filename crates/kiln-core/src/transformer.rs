//! The transformer forward pass.
//!
//! Real multi-head attention with Grouped Query Attention, RoPE,
//! causal masking, and full sequence processing. Produces coherent
//! logits from real weights.

use std::sync::Arc;

use kiln_kernels::{q4k_matmul_scalar, q6k_matmul_scalar};
use kiln_models::gguf::GgufFile;

use crate::chat::Forward;
use crate::transformer_config::TransformerConfig;
use crate::transformer_weights::{TransformerWeights, WeightError};

#[derive(Debug)]
pub enum TransformerError {
    Weight(WeightError),
    EmptyInput,
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformerError::Weight(e) => write!(f, "weight error: {}", e),
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

pub struct Transformer {
    config: TransformerConfig,
    weights: TransformerWeights,
}

fn decode_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 4;
        if off + 4 > bytes.len() { break; }
        let b = &bytes[off..off + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x { sum_sq += v * v; }
    let mean_sq = sum_sq / n as f32;
    let rms = (mean_sq + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for i in 0..n { out.push((x[i] / rms) * weight[i]); }
    out
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else {
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
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

fn dot_packed(weights: &[u8], num_weights: usize, x: &[f32]) -> f32 {
    let num_blocks = (num_weights + 255) / 256;
    let q4k_len = num_blocks * 144;
    let q6k_len = num_blocks * 210;
    let f32_len = num_weights * 4;
    let f16_len = num_weights * 2;
    if weights.len() == q4k_len {
        q4k_matmul_scalar(weights, num_weights, x)
    } else if weights.len() == q6k_len {
        q6k_matmul_scalar(weights, num_weights, x)
    } else if weights.len() == f32_len {
        let w = decode_f32(weights, num_weights);
        let mut acc = 0.0f32;
        for i in 0..num_weights { acc += w[i] * x[i]; }
        acc
    } else if weights.len() == f16_len {
        let mut acc = 0.0f32;
        for i in 0..num_weights {
            let b = &weights[i*2..i*2+2];
            let h = u16::from_le_bytes([b[0], b[1]]);
            acc += f16_to_f32(h) * x[i];
        }
        acc
    } else {
        0.0
    }
}

impl Transformer {
    pub fn from_gguf(file: &Arc<GgufFile>) -> Result<Self, TransformerError> {
        let weights = TransformerWeights::from_gguf(file)?;
        let config = weights.config.clone();
        Ok(Self { config, weights })
    }

    pub fn config_ref(&self) -> &TransformerConfig { &self.config }

    /// Project a single vector x through a packed weight matrix.
    /// Returns a vector of num_rows floats.
    fn proj(&self, weights: &[u8], num_rows: usize, num_cols: usize, x: &[f32]) -> Vec<f32> {
        if num_rows == 0 || weights.is_empty() { return Vec::new(); }
        let bytes_per_row = weights.len() / num_rows;
        let mut out = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let start = r * bytes_per_row;
            let end = start + bytes_per_row;
            if end > weights.len() { break; }
            out.push(dot_packed(&weights[start..end], num_cols, x));
        }
        out
    }

    /// Project a full sequence: shape [seq_len, in_dim] -> [seq_len, out_dim].
    fn proj_seq(&self, weights: &[u8], num_rows: usize, num_cols: usize, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * num_rows);
        for t in 0..seq_len {
            let x_t = &x[t * num_cols..(t + 1) * num_cols];
            let y_t = self.proj(weights, num_rows, num_cols, x_t);
            out.extend_from_slice(&y_t);
        }
        out
    }

    fn embed_token(&self, token: u32) -> Vec<f32> {
        use kiln_kernels::{q4k_dequant_block, q6k_dequant_block};
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let token = token as usize;
        if token >= vocab { return vec![0.0; hidden]; }
        let bytes = &self.weights.token_embd;
        if bytes.is_empty() { return vec![0.0; hidden]; }
        let bpr = bytes.len() / vocab;
        let start = token * bpr;
        let end = start + bpr;
        if end > bytes.len() { return vec![0.0; hidden]; }
        let row = &bytes[start..end];
        let num_blocks = (hidden + 255) / 256;
        let block_size = row.len() / num_blocks;
        let mut out = Vec::with_capacity(hidden);
        for b in 0..num_blocks {
            let bs = b * block_size;
            let be = bs + block_size;
            if be > row.len() { break; }
            let block = &row[bs..be];
            let base = b * 256;
            let count = (hidden - base).min(256);
            if block_size == 210 {
                let deq = q6k_dequant_block(block);
                out.extend_from_slice(&deq[..count]);
            } else if block_size == 144 {
                let deq = q4k_dequant_block(block);
                out.extend_from_slice(&deq[..count]);
            } else {
                out.extend(std::iter::repeat(0.0).take(count));
            }
        }
        out.resize(hidden, 0.0);
        out
    }

    /// Apply RoPE in place to Q or K.
    /// x shape: [seq_len, num_heads, head_dim].
    ///
    /// Qwen2.5 uses NeoX-style RoPE, also called the "rotate half"
    /// convention. The pairs are (d, d + head_dim/2), NOT (2d, 2d+1).
    /// Using the interleaved convention produces plausible but wrong
    /// attention scores, which is what we observed before this fix.
    fn apply_rope(&self, x: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize) {
        let base = self.config.rope_theta;
        let half = head_dim / 2;
        for t in 0..seq_len {
            let pos = t as f32;
            for h in 0..num_heads {
                let head_off = (t * num_heads + h) * head_dim;
                for d in 0..half {
                    // NeoX frequency: based on d, not 2d.
                    let freq = 1.0 / base.powf((2.0 * d as f32) / head_dim as f32);
                    let theta = pos * freq;
                    let c = theta.cos();
                    let s = theta.sin();
                    let x0 = x[head_off + d];
                    let x1 = x[head_off + d + half];
                    x[head_off + d] = x0 * c - x1 * s;
                    x[head_off + d + half] = x0 * s + x1 * c;
                }
            }
        }
    }

    pub fn forward_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let debug = std::env::var("KILN_DEBUG_TRANSFORMER").is_ok();
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let inter = self.config.intermediate_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;
        let seq_len = tokens.len();

        if seq_len == 0 { return vec![0.0; vocab]; }

        // Embed all tokens: [seq_len, hidden]
        let mut h = vec![0.0f32; seq_len * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            let e = self.embed_token(tok);
            h[t * hidden..(t + 1) * hidden].copy_from_slice(&e);
        }
        if debug {
            eprintln!("[dbg] embedding token 0 first 8: {:?}", &h[0..8]);
            eprintln!("[dbg] embedding token 0 sum: {}", h[0..hidden].iter().sum::<f32>());
            eprintln!("[dbg] embedding token 0 max abs: {}", h[0..hidden].iter().map(|v| v.abs()).fold(0.0f32, f32::max));
        }

        let group = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for layer in &self.weights.layers {
            // Pre-attention norm on every position
            let attn_norm_w = decode_f32(&layer.attn_norm, hidden);
            let mut normed = vec![0.0f32; seq_len * hidden];
            for t in 0..seq_len {
                let n = rms_norm(&h[t * hidden..(t + 1) * hidden], &attn_norm_w, eps);
                normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
            }
            if debug && std::sync::atomic::AtomicBool::new(true).load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[dbg] attn_norm_w first 8: {:?}", &attn_norm_w[0..8]);
                eprintln!("[dbg] normed first 8: {:?}", &normed[0..8]);
            }

            // Q, K, V projections for all positions
            let mut q = self.proj_seq(&layer.attn_q, q_dim, hidden, &normed, seq_len);
            let mut k = self.proj_seq(&layer.attn_k, kv_dim, hidden, &normed, seq_len);
            let mut v = self.proj_seq(&layer.attn_v, kv_dim, hidden, &normed, seq_len);

            // Add Q, K, V biases if present. Qwen2.5 uses them.
            // Without the biases, every attention score is offset by a
            // constant, which shifts the softmax and destroys attention.
            if !layer.attn_q_bias.is_empty() {
                let b = decode_f32(&layer.attn_q_bias, q_dim);
                for t in 0..seq_len {
                    let off = t * q_dim;
                    for i in 0..q_dim {
                        q[off + i] += b[i];
                    }
                }
            }
            if !layer.attn_k_bias.is_empty() {
                let b = decode_f32(&layer.attn_k_bias, kv_dim);
                for t in 0..seq_len {
                    let off = t * kv_dim;
                    for i in 0..kv_dim {
                        k[off + i] += b[i];
                    }
                }
            }
            if !layer.attn_v_bias.is_empty() {
                let b = decode_f32(&layer.attn_v_bias, kv_dim);
                for t in 0..seq_len {
                    let off = t * kv_dim;
                    for i in 0..kv_dim {
                        v[off + i] += b[i];
                    }
                }
            }

            // Apply RoPE to Q and K
            self.apply_rope(&mut q, seq_len, num_heads, head_dim);
            self.apply_rope(&mut k, seq_len, num_kv_heads, head_dim);

            // Attention output: [seq_len, q_dim]
            let mut attn_out = vec![0.0f32; seq_len * q_dim];
            for hq in 0..num_heads {
                let kvh = hq / group;
                for i in 0..seq_len {
                    // Compute scores against all j <= i
                    let mut scores = vec![0.0f32; i + 1];
                    let q_off = (i * num_heads + hq) * head_dim;
                    for j in 0..=i {
                        let k_off = (j * num_kv_heads + kvh) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[j] = dot * scale;
                    }
                    // Softmax over scores[0..i+1]
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    if sum > 0.0 { for e in exps.iter_mut() { *e /= sum; } }
                    // Weighted sum of V
                    let out_off = (i * num_heads + hq) * head_dim;
                    for j in 0..=i {
                        let v_off = (j * num_kv_heads + kvh) * head_dim;
                        let w = exps[j];
                        for d in 0..head_dim {
                            attn_out[out_off + d] += w * v[v_off + d];
                        }
                    }
                }
            }

            // Output projection: [seq_len, hidden]
            let attn_proj = self.proj_seq(&layer.attn_output, hidden, q_dim, &attn_out, seq_len);

            // Residual
            for i in 0..seq_len * hidden {
                h[i] += attn_proj[i];
            }

            // FFN norm
            let ffn_norm_w = decode_f32(&layer.ffn_norm, hidden);
            let mut ffn_normed = vec![0.0f32; seq_len * hidden];
            for t in 0..seq_len {
                let n = rms_norm(&h[t * hidden..(t + 1) * hidden], &ffn_norm_w, eps);
                ffn_normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
            }

            // Gate and up projections
            let gate = self.proj_seq(&layer.ffn_gate, inter, hidden, &ffn_normed, seq_len);
            let up = self.proj_seq(&layer.ffn_up, inter, hidden, &ffn_normed, seq_len);
            let mut act = vec![0.0f32; seq_len * inter];
            for i in 0..seq_len * inter {
                act[i] = silu(gate[i]) * up[i];
            }

            // Down projection
            let ffn_out = self.proj_seq(&layer.ffn_down, hidden, inter, &act, seq_len);

            // Residual
            for i in 0..seq_len * hidden {
                h[i] += ffn_out[i];
            }
        }

        // Final norm on the last position only
        let out_norm_w = decode_f32(&self.weights.output_norm, hidden);
        let last = &h[(seq_len - 1) * hidden..seq_len * hidden];
        let h_last = rms_norm(last, &out_norm_w, eps);

        // Output projection: [vocab, hidden] -> logits
        let output = &self.weights.output;
        let mut logits = vec![0.0f32; vocab];
        if output.is_empty() { return logits; }
        let bpr = output.len() / vocab;
        if bpr == 0 { return logits; }
        for t in 0..vocab {
            let start = t * bpr;
            let end = start + bpr;
            if end > output.len() { break; }
            logits[t] = dot_packed(&output[start..end], hidden, &h_last);
        }
        logits
    }
}

impl Forward for Transformer {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.forward_tokens(tokens)
    }
    fn vocab_size(&self) -> usize { self.config.vocab_size }
}
