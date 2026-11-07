# KILN quantization policy

**Status:** Active policy record under KILN-E35. The master roadmap remains authoritative for phase status and acceptance gates.

## Policy

GGUF structural recognition, tensor decoding, backend kernel coverage, and end-to-end model compatibility are separate claims. A file that parses is not necessarily executable by the active inference backend. Format support must be stated at the level actually demonstrated: parser, tensor operation, backend, or pinned model/workload.

KILN should retain compatibility with widely used GGUF formats while developing its TQ1_0 path. Ternary is an optimization and research direction; it is not a requirement imposed on users. No quantization format may be called generally supported based only on a design draft, converter, isolated decoder, or synthetic test.

## Current evidence

| Format or layer | Evidence/status | What may be claimed |
|---|---|---|
| GGUF container | The repository has GGUF discovery and structural validation. | Structural validation only; it does not certify architecture or runtime compatibility. |
| Q4_K | README records a historical Qwen2.5-1.5B Q4_K run on the Dell Latitude 7490. | This is a historical model-level result, not a current-head benchmark or universal Q4_K support guarantee. |
| TQ1_0 | The repository and roadmap record a native ternary path; current GGML type is 34 and packing is five trits per byte. | Identify the format/path as implemented; performance and quality claims still require their specific gates. |
| Other GGUF tensor types | Support depends on the pinned parser, tensor operations, and backend path. | Do not claim broad support without a verified support matrix and model-level evidence. |

The roadmap records measured and historical results with their conditions. Re-run the pinned Phase 3.5 benchmark before using old rates as current performance.

## Adding or changing a format

Before declaring a format supported, document:

1. GGUF type ID, block layout, alignment, effective bits/weight, scales and metadata.
2. Parser and validation behavior, including malformed/truncated input handling.
3. Implemented decode and compute operations, supported backends, and fallback behavior.
4. Differential correctness against a pinned reference implementation and representative tensors.
5. At least one pinned real-model load/inference result, with quality and performance reported separately.
6. Licensing/provenance and whether conversion is required.

Keep exact execution and lossy/approximate changes distinct. Any loss of precision or route substitution requires an explicit quality budget and evaluation. Never infer runtime performance from nominal bits per parameter alone.
