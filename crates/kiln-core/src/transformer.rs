//! The transformer forward pass.
//!
//! Real multi-head attention with Grouped Query Attention, RoPE,
//! causal masking, and full sequence processing. Produces coherent
//! logits from real weights.

use std::sync::Arc;

use crate::dispatch::{Dispatcher, Operation, ExecContext, Scratch, QuantKind};
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
    cache: crate::kv_cache::KvCache,
    dispatcher: crate::dispatch::Dispatcher,
}

fn decode_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    // Fail loudly. A short tensor is a bug in the model file or the
    // loader, never something to paper over.
    assert!(
        bytes.len() >= count * 4,
        "decode_f32: need {} bytes for {} floats, got {}",
        count * 4, count, bytes.len()
    );
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 4;
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

fn dot_packed(dispatcher: &Dispatcher, weights: &[u8], num_weights: usize, x: &[f32]) -> f32 {
    let kind = match QuantKind::detect(num_weights, weights.len()) {
        Some(k) => k,
        None => return 0.0,
    };
    let op = Operation::Matmul {
        weights: String::new(),
        quant: kind,
        out_features: 1,
        in_features: num_weights,
    };
    let mut scratch = Scratch::new();
    let mut out = [0.0f32; 1];
    let mut ctx = ExecContext {
        weight_bytes: weights,
        quant: kind,
        scratch: &mut scratch,
        out: &mut out,
        activations: x,
        seq_len: 1,
        position_offset: 0,
    };
    if dispatcher.dispatch(&op, &mut ctx).is_err() { return 0.0; }
    out[0]
}

#[allow(dead_code)]
fn dot_packed_legacy(weights: &[u8], num_weights: usize, x: &[f32]) -> f32 {
    let num_blocks = (num_weights + 255) / 256;
    let q4k_len = num_blocks * 144;
    let q6k_len = num_blocks * 210;
    let f32_len = num_weights * 4;
    let f16_len = num_weights * 2;
    if weights.len() == q4k_len {
        kiln_kernels::q4k_matmul_scalar(weights, num_weights, x)
    } else if weights.len() == q6k_len {
        kiln_kernels::q6k_matmul_scalar(weights, num_weights, x)
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
        let kv_dim = config.num_kv_heads * config.head_dim();
        let max_len = config.context_length.min(4096);
        let cache = crate::kv_cache::KvCache::new(config.num_layers, kv_dim, max_len);
        let mut dispatcher = crate::dispatch::Dispatcher::new();
        dispatcher.register(Box::new(crate::backends::cpu_ternary::CpuTernaryBackend::new()));
        dispatcher.register(Box::new(crate::backends::cpu_scalar::CpuScalarBackend::new()));
        Ok(Self { config, weights, cache, dispatcher })
    }

    pub fn config_ref(&self) -> &TransformerConfig { &self.config }

    /// Project a single vector x through a packed weight matrix.
    /// Returns a vector of num_rows floats.
    fn proj(&self, dispatcher: &Dispatcher, weights: &[u8], num_rows: usize, num_cols: usize, x: &[f32]) -> Vec<f32> {
        if num_rows == 0 || weights.is_empty() { return Vec::new(); }
        // The byte layout of a row is a property of the quant format.
        // Query it. Do not compute it here.
        let bytes_per_row = match QuantKind::detect(num_cols, weights.len() / num_rows) {
            Some(k) => k.row_bytes(num_cols),
            None => weights.len() / num_rows,  // fall back, will be caught by dot_packed
        };
        let mut out = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let start = r * bytes_per_row;
            let end = start + bytes_per_row;
            if end > weights.len() { break; }
            out.push(dot_packed(dispatcher, &weights[start..end], num_cols, x));
        }
        out
    }

    /// Project a full sequence: shape [seq_len, in_dim] -> [seq_len, out_dim].
    fn proj_seq(&self, dispatcher: &Dispatcher, weights: &[u8], num_rows: usize, num_cols: usize, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * num_rows);
        for t in 0..seq_len {
            let x_t = &x[t * num_cols..(t + 1) * num_cols];
            let y_t = self.proj(dispatcher, weights, num_rows, num_cols, x_t);
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
    fn apply_rope(&self, x: &mut [f32], pos_offset: usize, seq_len: usize, num_heads: usize, head_dim: usize) {
        let base = self.config.rope_theta;
        let half = head_dim / 2;
        for t in 0..seq_len {
            let pos = (pos_offset + t) as f32;
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

    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let inter = self.config.intermediate_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;

        if tokens.is_empty() { return vec![0.0; vocab]; }

        // Determine which tokens are new. If the caller's sequence starts
        // with what is already cached, only process the tail. Otherwise
        // reset and process everything.
        let cached = self.cache.len();
        let reuse = cached > 0
            && tokens.len() >= cached
            && tokens[..cached] == self.cache.cached_tokens[..];
        if !reuse { self.cache.clear(); }
        let pos_offset = if reuse { cached } else { 0 };
        let new_tokens = &tokens[pos_offset..];
        let n_new = new_tokens.len();
        if n_new == 0 {
            // Nothing to do. Re-run the last position's output projection
            // by reprocessing the final token alone.
            let last = *tokens.last().unwrap();
            self.cache.clear();
            return self.forward_tokens(&[last]);
        }
        if !self.cache.can_append(n_new) {
            panic!("KV cache overflow: cached {} + new {} > max {}",
                cached, n_new, self.cache.max_len);
        }

        // Embed only the new tokens: [n_new, hidden]
        let mut h = vec![0.0f32; n_new * hidden];
        for (t, &tok) in new_tokens.iter().enumerate() {
            let e = self.embed_token(tok);
            h[t * hidden..(t + 1) * hidden].copy_from_slice(&e);
        }

        let group = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for (li, layer) in self.weights.layers.iter().enumerate() {
            // Pre-attention norm on new positions
            let attn_norm_w = decode_f32(&layer.attn_norm, hidden);
            let mut normed = vec![0.0f32; n_new * hidden];
            for t in 0..n_new {
                let n = rms_norm(&h[t * hidden..(t + 1) * hidden], &attn_norm_w, eps);
                normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
            }

            // Q, K, V projections for new positions only
            let mut q = self.proj_seq(&self.dispatcher, &layer.attn_q, q_dim, hidden, &normed, n_new);
            let mut k = self.proj_seq(&self.dispatcher, &layer.attn_k, kv_dim, hidden, &normed, n_new);
            let mut v = self.proj_seq(&self.dispatcher, &layer.attn_v, kv_dim, hidden, &normed, n_new);

            if !layer.attn_q_bias.is_empty() {
                let b = decode_f32(&layer.attn_q_bias, q_dim);
                for t in 0..n_new { let off = t * q_dim;
                    for i in 0..q_dim { q[off + i] += b[i]; } }
            }
            if !layer.attn_k_bias.is_empty() {
                let b = decode_f32(&layer.attn_k_bias, kv_dim);
                for t in 0..n_new { let off = t * kv_dim;
                    for i in 0..kv_dim { k[off + i] += b[i]; } }
            }
            if !layer.attn_v_bias.is_empty() {
                let b = decode_f32(&layer.attn_v_bias, kv_dim);
                for t in 0..n_new { let off = t * kv_dim;
                    for i in 0..kv_dim { v[off + i] += b[i]; } }
            }

            // RoPE with absolute positions
            self.apply_rope(&mut q, pos_offset, n_new, num_heads, head_dim);
            self.apply_rope(&mut k, pos_offset, n_new, num_kv_heads, head_dim);

            // Append fresh K, V to the cache
            for t in 0..n_new {
                let ko = t * kv_dim;
                let vo = t * kv_dim;
                self.cache.layers[li].push(&k[ko..ko + kv_dim], &v[vo..vo + kv_dim]);
            }

            let total_len = pos_offset + n_new;

            // Attention: query i (absolute pos = pos_offset + i) attends
            // to all cached positions 0..=pos_offset+i.
            let mut attn_out = vec![0.0f32; n_new * q_dim];
            for hq in 0..num_heads {
                let kvh = hq / group;
                for i in 0..n_new {
                    let abs_i = pos_offset + i;
                    let q_off = (i * num_heads + hq) * head_dim;
                    let mut scores = vec![0.0f32; abs_i + 1];
                    for j in 0..=abs_i {
                        let k_row = self.cache.k_row(li, j);
                        let k_off = kvh * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k_row[k_off + d];
                        }
                        scores[j] = dot * scale;
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    if sum > 0.0 { for e in exps.iter_mut() { *e /= sum; } }
                    let out_off = (i * num_heads + hq) * head_dim;
                    for j in 0..=abs_i {
                        let v_row = self.cache.v_row(li, j);
                        let v_off = kvh * head_dim;
                        let w = exps[j];
                        for d in 0..head_dim {
                            attn_out[out_off + d] += w * v_row[v_off + d];
                        }
                    }
                }
            }

            // Output projection on new positions only
            let attn_proj = self.proj_seq(&self.dispatcher, &layer.attn_output, hidden, q_dim, &attn_out, n_new);
            for i in 0..n_new * hidden { h[i] += attn_proj[i]; }

            // FFN on new positions
            let ffn_norm_w = decode_f32(&layer.ffn_norm, hidden);
            let mut ffn_normed = vec![0.0f32; n_new * hidden];
            for t in 0..n_new {
                let n = rms_norm(&h[t * hidden..(t + 1) * hidden], &ffn_norm_w, eps);
                ffn_normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
            }
            let gate = self.proj_seq(&self.dispatcher, &layer.ffn_gate, inter, hidden, &ffn_normed, n_new);
            let up = self.proj_seq(&self.dispatcher, &layer.ffn_up, inter, hidden, &ffn_normed, n_new);
            let mut act = vec![0.0f32; n_new * inter];
            for i in 0..n_new * inter { act[i] = silu(gate[i]) * up[i]; }
            let ffn_out = self.proj_seq(&self.dispatcher, &layer.ffn_down, hidden, inter, &act, n_new);
            for i in 0..n_new * hidden { h[i] += ffn_out[i]; }

            let _ = total_len;
        }

        // Record the new tokens
        self.cache.append_tokens(new_tokens);

        // Final norm on the last new position only
        let out_norm_w = decode_f32(&self.weights.output_norm, hidden);
        let last = &h[(n_new - 1) * hidden..n_new * hidden];
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
            logits[t] = dot_packed(&self.dispatcher, &output[start..end], hidden, &h_last);
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
