# AVX2 Fused Matmul — Design

## Why

Current Q4_K and Q6_K matmul kernels in cpp/kernels/q4k.c and q6k.c are
scalar C. Each block is dequantized to a 256-float stack buffer, then
dot-producted against the activations with a scalar loop. Measured on
Dell 7490: 20-token reply takes 48 s, 0.41 tok/s. The kernel is the
floor.

AVX2 is available on every x86_64 target KILN supports, down to Tier 0.
The target is 4-8x on the fused matmul. No change to the on-disk
format, no change to the Rust API, no change to the model file.

## Non-goals

- Not AVX-512. Tier 0 has no AVX-512.
- Not a new quantization format.
- Not a rewrite of the pack or dequant paths that already work.
- Not multithreading. That is a later design.

## Existing C ABI (unchanged)

  int kiln_q4k_matmul_scalar(const uint8_t*, size_t, const float*, size_t, float*);
  int kiln_q6k_matmul_scalar(const uint8_t*, size_t, const float*, size_t, float*);

New functions with the same signatures:

  int kiln_q4k_matmul_avx2(...);
  int kiln_q6k_matmul_avx2(...);

## Dispatch

A new function selects the path at runtime based on CPUID:

  int kiln_q4k_matmul(const uint8_t*, size_t, const float*, size_t, float*);

The Rust FFI binds kiln_q4k_matmul (not the scalar or avx2 variant).
Dispatch happens once per call via a cached atomic bool.

Detection via __builtin_cpu_supports("avx2") on gcc/clang. On non-x86
targets the scalar path is the only path, compiled without -mavx2.

## Algorithm — Q4_K

For each 256-weight superblock:
  1. Read d, dmin (two f16).
  2. Unpack the 8 sub-block scales and 8 mins.
  3. For each of 4 groups of 32 bytes:
       a. Load 32 bytes with _mm256_loadu_si256.
       b. Split into low nibbles and high nibbles.
       c. Convert to f32 using the 8-bit LUT trick: precompute
          scale_lut[16] = {0*s, 1*s, ..., 15*s} for each scale.
       d. Multiply by activations[g*64 .. g*64+32] and
          activations[g*64+32 .. g*64+64], accumulate.
  4. Subtract the min contributions: sum(activation) * min for the
     block.
  5. Add to running scalar accumulator.

The dequantized weights are never materialized. The dot product is
computed directly from the nibbles.

## Algorithm — Q6_K

Same structure. Each 128-weight half is processed as 4 groups of 32
outputs. The 6-bit values are assembled from ql and qh with shifts
before LUT lookup. The LUT has 64 entries (-32..31) scaled by the
int8 sub-block scale.

## Register budget

AVX2 has 16 ymm registers. Q4_K inner loop uses:
  2 for the byte loads
  2 for the low/high nibble masks
  4 for the LUTs (two scales per group)
  2 for the activations
  2 for accumulators
Leaves ~4 for temporaries. Fits without spilling.

## Testing

Test 1 — differential. For N random Q4_K blocks and N random Q6_K
blocks, avx2 output must equal scalar output bit-for-bit within 1e-6
relative error. This is the anchor.

Test 2 — real block. A real superblock from the Qwen2.5-1.5B GGUF.
Same assertion.

Test 3 — dispatch. kiln_q4k_matmul must call the avx2 path on AVX2
hardware. Verified by printing the path once under a debug env var.

## Wiring in Rust

transformer.rs dot_packed currently calls q4k_matmul_scalar /
q6k_matmul_scalar directly. The wiring changes to call the dispatch
functions instead. One-line change in dot_packed.

kiln-kernels' build.rs already sets -mavx2 unconditionally on x86.
The avx2 functions use intrinsics guarded by #ifdef __AVX2__. On
targets without AVX2, only the scalar path compiles.

## Expected performance

Per the llama.cpp and T-MAC benchmarks, AVX2 LUT kernels for Q4_K
typically achieve 4-8x over scalar on Skylake-class CPUs. Dell 7490
i7-8650U is Kaby Lake (same core, marginally lower clocks).

Target: 20-token reply in <15 s, >1.3 tok/s. Sustained after throttling:
maybe 12-15 s, ~1.5 tok/s.

## Sequencing

  1. Write AVX2 Q4_K kernel + dispatch + differential test.
  2. Verify on Dell 7490.
  3. Write AVX2 Q6_K kernel + differential test.
  4. Verify.
  5. Wire dot_packed to the dispatch functions.
  6. Chat benchmark. Record the number.
  7. Rule KILN-E34 docs update.

## Anti-patterns

- Do not materialize dequantized weights into a scratch buffer. That
  is what the scalar path already does and it is the bottleneck.
- Do not skip the differential test. It is the anchor.
- Do not change the C ABI. The Rust FFI must not need to change
  beyond the dispatch name.
- Do not use FMA unless __FMA__ is defined. Not all AVX2 CPUs have it.
- Do not assume unaligned loads are free. Use loadu, not load.
