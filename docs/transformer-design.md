# KILN Transformer Forward Pass Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for the transformer forward pass. Anchors the
piece that turns real weights into real logits.

## Non-goals

This document does not specify multi-head attention. It does not specify
RoPE or any positional encoding. It does not specify the KV cache. It
specifies a minimal single-head transformer that exercises every code
path, produces deterministic logits, and can be extended to the full
model later.

## Why this shape

A full transformer is a large piece of code. If we write it all at once
and it produces wrong output, finding the bug is hard. If we write a
minimal version first, every piece is testable in isolation. Then we
add complexity one piece at a time, with differential tests against
the minimal version.

The minimal version is:

- Token embedding lookup (no positional encoding)
- RMSNorm
- Single-head attention
- SwiGLU feed-forward
- Residual connections
- Final RMSNorm
- Output projection to vocabulary logits

What it does NOT have yet:

- Multi-head attention
- RoPE or any positional encoding
- KV cache
- Grouped-query attention
- Sliding window attention
- Any optimization specific to a model family

The minimal version will produce low-quality output because it lacks
positional information and multiple heads. That is expected. The point
is to prove that real weights flow through real math and produce
deterministic logits.

## Where it lives

The transformer lives in kiln-core, module transformer.rs. It
implements the Forward trait from kiln-core::chat.

  pub struct Transformer {
      weights: TransformerWeights,
      config: TransformerConfig,
  }

  impl Forward for Transformer {
      fn forward(&mut self, tokens: &[u32]) -> Vec<f32>;
      fn vocab_size(&self) -> usize;
  }

The chat loop calls forward. The transformer returns logits. The
sampler picks a token. Nothing in the chat loop changes.

## The configuration

  pub struct TransformerConfig {
      pub vocab_size: usize,
      pub hidden_size: usize,
      pub num_layers: usize,
      pub intermediate_size: usize,
      pub num_heads: usize,       // currently 1
      pub rms_norm_eps: f32,
      pub context_length: usize,
  }

The config is read from GGUF metadata keys:

  general.architecture         string, "llama", "qwen2", etc.
  <arch>.vocab_size            u32
  <arch>.embedding_length      u32, the hidden size
  <arch>.block_count           u32, the number of layers
  <arch>.feed_forward_length   u32, the intermediate size
  <arch>.attention.head_count  u32, the number of heads
  <arch>.attention.layer_norm_rms_epsilon  f32

The <arch> prefix is the model family. For Llama it is "llama". For
Qwen it is "qwen2". The config reader strips the prefix.

## The weights

  pub struct TransformerWeights {
      pub token_embd: Vec<f32>,        // [vocab_size, hidden_size]
      pub output_norm: Vec<f32>,       // [hidden_size]
      pub output: Vec<f32>,            // [vocab_size, hidden_size]
      pub layers: Vec<LayerWeights>,
  }

  pub struct LayerWeights {
      pub attn_norm: Vec<f32>,         // [hidden_size]
      pub attn_q: Vec<f32>,            // [hidden_size, hidden_size]
      pub attn_k: Vec<f32>,            // [hidden_size, hidden_size]
      pub attn_v: Vec<f32>,            // [hidden_size, hidden_size]
      pub attn_output: Vec<f32>,       // [hidden_size, hidden_size]
      pub ffn_norm: Vec<f32>,          // [hidden_size]
      pub ffn_gate: Vec<f32>,          // [intermediate_size, hidden_size]
      pub ffn_up: Vec<f32>,            // [intermediate_size, hidden_size]
      pub ffn_down: Vec<f32>,          // [hidden_size, intermediate_size]
  }

Each weight is a flat Vec<f32>. The shape is encoded in the GGUF tensor
metadata. The loader returns the packed bytes. The transformer decodes
them to f32 using the tensor's dtype (F32, F16, or Tq1_0).

Decoding TQ1_0 weights to f32 uses the unpack function from
kiln-kernels. That is the path that connects the ternary kernel to the
transformer.

## The forward pass

  Step 1. Token embedding. For each token in the sequence, look up
          its embedding vector from token_embd. The result is a
          sequence of hidden_size vectors.

  Step 2. For each layer in layers:
    Step 2a. RMSNorm the input. Save the result.
    Step 2b. Attention. Compute Q, K, V from the normed input.
             Compute the attention scores, apply softmax, apply to V.
             For single-head, no head splitting.
    Step 2c. Attention output projection.
    Step 2d. Residual add.
    Step 2e. RMSNorm.
    Step 2f. SwiGLU feed-forward. Compute gate and up projections,
             apply SiLU to gate, multiply by up, project down.
    Step 2g. Residual add.

  Step 3. Final RMSNorm.

  Step 4. Output projection. Multiply the last token's hidden state by
          the output weight matrix. The result is a logits vector of
          length vocab_size.

  Step 5. Return the logits.

The forward pass processes the whole token sequence at once. It does
not cache anything between calls. Every call re-runs the whole
sequence. That is O(n^2) in sequence length but correct. The KV cache
is Phase 2.

## The math primitives

Four functions are needed:

  fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32>;
  fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32>;
  fn softmax(x: &[f32]) -> Vec<f32>;
  fn silu(x: f32) -> f32;

Each one is simple. Each one has a unit test.

matvec is the hot path. It is called dozens of times per forward pass.
For the first version it uses the scalar fused matmul kernel when the
weight is TQ1_0. For F32 and F16 weights, it uses a plain scalar loop.
Optimization comes later.

## Error handling

  pub enum TransformerError {
      MissingWeight(String),
      WrongShape { tensor: String, expected: Vec<u64>, got: Vec<u64> },
      UnsupportedDtype(String),
      Metadata(String),
  }

Each error names the tensor and the expected shape.

## Testing strategy

Three tests:

  1. Unit tests for the four math primitives. Hand-computed values.

  2. A tiny transformer with known weights. Hand-compute the expected
     output for one token. Verify the forward pass produces it.

  3. Determinism. Run the same input twice. Verify identical logits.

The tiny transformer uses a vocabulary of 4 tokens, hidden size 8,
one layer, and an intermediate size of 16. The weights are small
enough to write by hand in the test.

## The synthetic transformer GGUF

To test the transformer without a real model, we build a synthetic
GGUF with all the transformer weights and metadata. The Python script
that builds it writes the tensor table and the weight data with the
same layout the loader expects.

The synthetic transformer uses the tiny config above. Every weight
tensor is named according to the GGUF convention:

  token_embd.weight
  output_norm.weight
  output.weight
  blk.0.attn_norm.weight
  blk.0.attn_q.weight
  blk.0.attn_k.weight
  blk.0.attn_v.weight
  blk.0.attn_output.weight
  blk.0.ffn_norm.weight
  blk.0.ffn_gate.weight
  blk.0.ffn_up.weight
  blk.0.ffn_down.weight

That is the real GGUF naming for Llama-family models. Loading these
from a synthetic file proves the transformer works. Loading them from
a real model comes next.

## What comes after

Multi-head attention. RoPE. KV cache. Grouped-query attention. Each
one is a separate design document and a separate commit. The minimal
transformer stays as the reference implementation.

## Open questions

  - Should the weights be stored as f32 in memory, or kept packed and
    unpacked on the fly? Memory says keep packed. Speed says unpack
    once at load time. For a 7B model at 1.58 bits, unpacking to f32
    is a 20x memory increase. That does not fit in 16GB. The weights
    must stay packed.
  - How is the matvec implemented when the weight is packed TQ1_0?
    The fused matmul kernel is the answer. The transformer calls the
    fused kernel with the packed weight row and the activation vector.
    That path is already built.
  - Does the transformer run on the CPU backend only? Yes for Phase 1.
    GPU backends come in Phase 4.
  - How does the transformer report per-layer timing? The scheduler
    already records per-node cost. The transformer integrates with
    that by exposing layer boundaries as nodes. For the first version,
    the transformer does not integrate with the scheduler. It runs as
    one monolithic function. Integration comes when the scheduler can
    dispatch into the transformer.

## Version history

2026-10-03. Initial draft. Minimal single-head transformer. Config from
GGUF metadata. Weights stay packed. Four math primitives. Three tests.
Synthetic transformer GGUF for testing. Four open questions.
