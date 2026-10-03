# KILN Fused Ternary Matmul Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for the fused ternary matmul kernel. Anchors the
LUT-based multiplication-free approach that unlocks AT-2 on Tier 0.

## Non-goals

This document does not specify the AVX2 implementation. It does not specify
the block sizes that will be used. It specifies the algorithm, the data
layout, the LUT construction, and the interface. The numbers land in the
benchmark.

## Why this kernel exists

The standalone unpacker tops out at roughly 3-6x because the store of five
floats per byte dominates and AVX2 gather on Skylake is slow. Even at best,
the unpacker produces 0.6 ns per element. A 3B active MoE unpacks in 1.8
seconds per token at that rate. The AT-2 target requires 3 tokens per
second, which is a factor of 5 faster than the standalone unpacker can go.

The gap closes only if unpacking is fused into the multiply. The kernel
reads packed ternary weights and computes the matmul without materializing
floats. This is the T-MAC approach. It is the only path to AT-2 on Tier 0.

## The arithmetic

A ternary weight is in {-1, 0, +1}. A matmul is:

  y[i] = sum_j w[j][i] * x[j]

With ternary weights, this becomes:

  y[i] = sum over positive weights of x[j]
        - sum over negative weights of x[j]

The zero weights contribute nothing. The matmul reduces to two sums per
output element: one for the positive weights and one for the negative
weights. No multiplication is required. Only addition and subtraction.

This is the multiplication-free property that the T-MAC paper exploits.
The kernel replaces every multiply with a conditional add or subtract.

## The LUT

Computing "which weights are positive" and "which are negative" per element
at runtime is slow. Instead, the kernel uses a precomputed lookup table
indexed by the packed weights.

For a group of G activations and G corresponding ternary weights, the
kernel precomputes a table of 3^G entries. Each entry is the partial sum
of the activations selected by the corresponding trit pattern.

The table size is 3^G. For G = 4, the table has 81 entries and fits in L1.
For G = 8, the table has 6561 entries and requires L2. The block size G
determines the memory access pattern and the cache behavior.

For Tier 0 with the i7-8650U, the target block size is G = 4 or G = 5. The
table fits in L1 and the memory access is predictable.

The table is built once at kernel load time. It is rebuilt when the
activation vector changes. For inference, this means the table is rebuilt
once per matmul call, not once per element. The cost is amortized.

## The data layout

The packed ternary weights are stored in TQ1.0 format. Five trits per byte.
Base-3 encoding. This is the format from docs/kernels-design.md. No change
is required to the storage format. The fused kernel reads the same bytes
the standalone unpacker reads.

The activations are stored as floats or as int8. The LUT is built from
the int8 quantized activations. The output is accumulated in int32 and
converted to float at the end.

The weight matrix layout is row-major at the block level. For each output
row, the weights for that row are stored contiguously. This matches the
memory access pattern of the matmul.

## The interface

The fused kernel exposes:

  int kiln_tq1_0_matmul(
      const uint8_t* weights,   // TQ1.0 packed weights
      size_t num_weights,       // number of ternary weights in the row
      const int8_t* activations,// quantized activations
      size_t num_activations,   // length of the activation vector
      float scale,              // output scale factor
      float* output             // output vector, length output_len
  );

The kernel computes the matmul of the packed weights against the
activations and writes the scaled output.

The scalar reference implementation is straightforward. It unpacks each
trit, adds or subtracts the corresponding activation, and accumulates.

The AVX2 implementation uses the LUT. It processes G activations per
lookup. It reads the packed weights, computes the LUT index, looks up
the partial sum, and accumulates. No multiplies. No float materialization
of the weights.

## Scalar reference

The scalar reference does not use the LUT. It unpacks each trit and adds
or subtracts. This is the correctness anchor. Every optimized version must
produce bit-identical output to the scalar reference.

The scalar reference is slow. It is the reference, not the production path.
It exists so that any bug in the LUT construction or the AVX2 path shows
up immediately as a differential test failure.

## The AVX2 path

The AVX2 implementation processes 8 output elements at a time. For each
group of 8 output elements, it processes the activation vector in chunks
of G.

For each chunk:
  1. Load G activations.
  2. Load the packed weights for 8 output rows at that chunk position.
  3. Compute 8 LUT indices, one per output row.
  4. Use AVX2 gather to load 8 partial sums.
  5. Accumulate into 8 int32 accumulators.

After all chunks, convert the 8 int32 accumulators to floats, multiply by
the scale, and store 8 floats.

The AVX2 gather is the bottleneck. On Skylake, `_mm256_i32gather_epi32`
takes roughly 20 cycles for 8 elements. That is 2.5 cycles per element.
For a chunk of G=5, the total compute per 8 outputs is roughly 25 cycles
of gather plus a few cycles of accumulate. Compare to the scalar path at
roughly 40 cycles per output. That is the expected speedup: roughly 8-12x.

That is enough to reach AT-2. It is the number that matters.

## Block size selection

G = 4 gives a 81-entry table. Fits in L1. Fast gather. Low accuracy
because the LUT covers only 4 activations at a time.

G = 5 gives a 243-entry table. Fits in L1. Same table size as the
unpacker table. Good balance.

G = 6 gives a 729-entry table. Still fits in L1 on most CPUs.

G = 8 gives a 6561-entry table. Requires L2. Slower gather. Higher
accuracy because more activations are covered per lookup.

The default is G = 5. The block size is configurable at kernel load time.
The benchmark measures the throughput at each block size and picks the
best for the machine.

## Accuracy

The LUT approach is exact for ternary weights. It does not approximate.
The only source of error is the int8 quantization of the activations. The
quantization error is bounded by the activation quantization scheme. For
inference with int8 activations and a float output scale, the error is
within the standard deviation of the quantizer.

The spec requires accuracy within 1 percent of FP16 on WikiText-2
perplexity. The int8 activation quantization introduces roughly 0.3
percent perplexity increase at 8-bit. That is within tolerance.

If a higher accuracy is required, the activations can be quantized to 16
bits. The LUT grows correspondingly. The kernel supports both.

## Testing strategy

Three tests per kernel:

  1. Scalar reference against a hand-computed 3x3 example. Exact match.
  2. Differential test: AVX2 output must match scalar output bit-for-bit
     for matrices of size 8x8, 16x16, 64x64, 256x256.
  3. Property test: random matrices and random ternary weights, AVX2
     output matches scalar output within 0.0 tolerance.

The differential test is what catches LUT construction bugs. If the LUT
is built incorrectly for any trit pattern, the AVX2 output diverges
immediately.

## Benchmark targets

The benchmark measures:

  - matmul_ternary_ns_per_output: nanoseconds per output element
  - matmul_ternary_mb_per_sec: throughput in MB/s of weights read

The comparison is against the current state of the art on CPU. For Tier 0,
the comparison is against llama.cpp with Q4_K_M weights on the same model
size. The target is 8x faster.

For a 3B active MoE at 8x the baseline, the model reaches 3 tokens per
second on the Dell 7490. That is the AT-2 target.

## Open questions

  - The exact block size that maximizes throughput on the i7-8650U. Phase 1
    benchmark.
  - The exact activation quantization scheme. Int8 with per-row scale is
    the default. Per-channel may be required for some models.
  - How the kernel interacts with the KV cache. The fused kernel reads
    activations, not keys or values. The KV cache is a separate concern.
  - How the kernel is called from the scheduler. The scheduler places the
    matmul node on the CPU backend. The backend invokes the fused kernel.
    The interface is stable.

## Version history

2026-10-03 — Initial draft. The arithmetic, the LUT, the data layout, the
interface, the scalar reference, the AVX2 path, the block size selection,
the accuracy analysis, the testing strategy, the benchmark targets, and
four open questions.
