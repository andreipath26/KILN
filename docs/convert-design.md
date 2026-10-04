# kiln convert — Design

## Purpose

Phase 2 gate requires a ternary version of Qwen2.5-1.5B to measure the
2x target against. No prebuilt TQ1.0 GGUF exists. We build a converter.

`kiln convert --input <src.gguf> --output <dst.gguf> --quant tq1_0`

## Scope

Phase 2 minimum: Q4_K and Q6_K input tensors, TQ1.0 output. f32 and
f16 input also accepted (they are already the reference).

Out of scope for now: MoE expert layout, per-tensor mixed precision,
the UMF container format. Those are Phase 4.

## Algorithm

1. Open the source GGUF. Read header, metadata, tensor table.
2. Copy the metadata verbatim into the destination. Update
   `general.file_type` and any `*.quantization` keys to indicate
   TQ1.0.
3. For each tensor:
   a. Read the tensor's current quant type.
   b. Dequant the bytes to a Vec<f32> of length numel.
   c. Ternarize each value: -1.0, 0.0, or +1.0. Nearest of the three
      by absolute distance. Optionally clamp: values with |v| below
      a threshold become 0.
   d. Pack with kiln_kernels::pack.
   e. Emit the tensor with GGML type TQ1_0.
4. Write the new GGUF.

## Ternarization rule

For each f32 value v:

  if v < -0.5: -1.0
  elif v > 0.5: +1.0
  else: 0.0

The 0.5 threshold is the natural midpoint. AYOT (roadmap Pillar 2)
uses a learned threshold per tensor; Phase 2 uses the fixed 0.5
threshold as the baseline. Learned thresholds are Phase 2.5.

## GGML type for TQ1.0

GGUF uses GGML type ids. TQ1_0 is not yet a registered ggml type. We
use a reserved id in the experimental range. Document the id in the
converter so KILN and any future tool read the same file.

For Phase 2, choose id 200. It is unassigned in current ggml.

## Files

New:
- crates/kiln-models/src/convert.rs — the conversion pass.
- crates/kiln-models/src/gguf_write.rs — GGUF writer.
- crates/kiln-cli/src/cmd/convert.rs — CLI surface.

Modified:
- crates/kiln-models/src/lib.rs — expose the modules.
- crates/kiln-cli/src/main.rs — add the Convert subcommand.

## Tests

Test 1 — roundtrip F32. Convert a tiny F32 tensor to TQ1.0, read it
back, verify the trits match the ternarization rule applied to the
original.

Test 2 — Q4_K input. Convert one Q4_K block, verify values.

Test 3 — end to end. Convert Qwen2.5-1.5B, then run kiln debug on the
output. Top-1 must still be Paris. Perplexity on WikiText-2 within 2%
of the Q4_K baseline.

## Anti-patterns

- Do not stream. Convert loads one tensor at a time; the model is
  986 MB; RSS stays under 4 GB.
- Do not use an mmap write. Write the file once, sequentially.
- Do not touch the tokenizer or the chat template. Those are
  metadata and copy verbatim.
- Do not skip Test 3. Top-1 Paris on the converted model is the
  acceptance criterion for Phase 2.

## Gate 2

kiln convert --input models/tiny/qwen25-1.5b.gguf --output \
  models/tiny/qwen25-1.5b-tq1.0.gguf --quant tq1_0

produces a model file that:
- loads in kiln debug
- gives Paris as top-1 for "The capital of France is"
- runs 2x faster than the Q4_K baseline on the same prompt
- has WikiText-2 perplexity within 2% of Q4_K
