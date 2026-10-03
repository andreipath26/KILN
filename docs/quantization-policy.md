# KILN Quantization Policy

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Record the design decision about which quantization formats
KILN supports natively.

## The problem

The masses bring models they already have. Those models are almost
never in TQ1_0. They are in the standard GGUF quantizations that the
ecosystem uses: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32. If KILN only
supports TQ1_0, it fails the vast majority of real models.

The inspect of a real Qwen2.5 1.5B model proved it:

  168 tensors at dtype code 12 (Q4_K)
   29 tensors at dtype code 14 (Q6_K)
  141 tensors at F32

If KILN cannot decode Q4_K and Q6_K, it cannot run this model. This
is a 986 MB model that a user already has. Rejecting it is not
acceptable.

## The decision

KILN supports the standard GGUF quantizations as first-class formats.
Ternary is an optimization, not a requirement.

Natively supported in Phase 1:

  F32        32-bit float, uncompressed
  F16        16-bit float, uncompressed
  Q4_0       4-bit, group size 32, legacy
  Q4_K       4-bit, group size 32, K-quant, modern
  Q5_K       5-bit, group size 32, K-quant, modern
  Q6_K       6-bit, group size 32, K-quant, modern
  Q8_0       8-bit, group size 32, legacy
  TQ1_0      KILN ternary, 5 trits per byte

Deferred to Phase 2:

  Q2_K, Q3_K, Q3_K_S, Q3_K_M, Q3_K_L
  IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M
  IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M
  IQ4_XS, IQ4_NL

The deferred formats are lower priority because they are less common.
They will be added if users need them.

## Why Q4_K and Q6_K matter most

They are the two most common quantizations in the ecosystem. Ollama,
LM Studio, llama.cpp, and HuggingFace all default to Q4_K_M for
distribution. A user who downloads a model from HuggingFace gets a
Q4_K_M file. That file must load.

## The kernel strategy

Each supported format needs a decode function and a matmul kernel.

The decode function takes packed bytes and returns f32 values. It is
used for norms, biases, and any tensor where the caller needs the
actual float values.

The matmul kernel takes packed bytes and an activation vector and
returns the dot product. It does not materialize the weights. It is
the hot path for inference.

For the first version, only the decode functions are needed. The
matmul uses decode-then-matvec. That is slower than a fused kernel
but correct. The fused kernels come later, in priority order:

  1. Q4_K fused matmul  (most common format)
  2. Q6_K fused matmul
  3. Q8_0 fused matmul
  4. TQ1_0 fused matmul (already exists, scalar only)

## What this means for the roadmap

The Phase 1 transformer can load a real Q4_K model once the decode
functions exist. That is the fastest path to a working assistant.

The Phase 2 SSD streaming and MoE work still applies. Those optimizations
work on top of whatever quantization the model uses.

The TQ1_0 ternary path remains the KILN-specific speed advantage. It is
what makes the smallest, slowest machines fast enough. But it is an
optimization, not a gate.

## Rule

Rule KILN-E35. KILN supports the standard GGUF quantizations natively.
Ternary is an optimization. No user is excluded because their model is
in a different format.
