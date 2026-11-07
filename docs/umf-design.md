# Unified Model Format (UMF) — design and implementation status

## Authority and current status

This is the Phase 5.1 design record. **The embedded KUMF container described below is a proposal, not the format implemented on main.** Current code provides a separate expert sidecar beside an unchanged GGUF.

Merged PRs #41–44 add:

- explicit GGUF tensor-to-expert component maps;
- a separate, 4 KiB-aligned expert blob and index;
- per-expert CRC32 integrity checks and verified component reads;
- bounded copy/read paths that preserve the original GGUF bytes.

The mapping is supplied explicitly and validated against the GGUF. Automatic decomposition of fused expert matrices is not generally implemented. CRC32 detects accidental corruption; it is not an authenticity mechanism. KILN does not yet materialize those sidecar components into llama.cpp's live MoE compute graph. See docs/phase5-streaming.md for the current inference integration boundary.

## Current sidecar artifact

The current workflow leaves model.gguf unchanged and writes separate files, typically experts.bin, experts.index.json, experts.crc32.json, and experts.components.json. This preserves the ordinary GGUF input path while enabling KILN's own bounded expert-byte readers. The exact file set depends on whether packing uses an explicit expert-range table or named tensor components.

This is not a single-file UMF extension, does not embed prerouter weights or cache policy, and does not claim that stock llama.cpp consumes the sidecar. A sidecar round-trip test proves copied bytes and metadata integrity for the supplied map; it does not prove that a model's map is complete or that the inference runtime uses the sidecar.

## Proposed embedded KUMF container (not implemented)

The original proposal is a GGUF file plus a KILN extension section. The proposed layout is:

1. unchanged GGUF header, metadata and tensor table;
2. ordinary GGUF tensor data;
3. KILN extension header (proposed magic KUMF, version 1);
4. expert manifest and expert-contiguous data;
5. optional prerouter weights and cache metadata;
6. optional per-tier tuning hints.

No parser/writer for this embedded extension is currently established by the sidecar implementation. Do not treat the proposed layout or its compatibility statements as tested behavior.

## Proposed runtime metadata (not implemented)

The earlier design proposed per-tier tuning hints such as sustained-throughput fraction, throttle temperature and recommended thread count. No evidence currently shows those fields being emitted into or consumed from a KUMF artifact. Keep thermal policy in the runtime design until a concrete format and calibration experiment justify persisting it in a model artifact.

## Compatibility

- The sidecar path leaves the source GGUF unchanged; a compatible stock GGUF reader can continue to load that original file.
- Stock llama.cpp does not consume the separate KILN expert sidecar.
- A plain GGUF remains the ordinary KILN/llama.cpp model input.
- The original proposal that stock llama.cpp will ignore an embedded KUMF extension has not been established by an implemented KUMF parser/writer and compatibility test.
- Exact sidecar support depends on a complete, validated model-specific mapping and a KILN runtime path that consumes the resulting components.

## Proposed embedded-container build process (not implemented)

The earlier design proposed that kiln convert could:

1. read a source GGUF;
2. extract expert tensors into expert-contiguous storage;
3. write a KUMF manifest and extension;
4. optionally write trained prerouter and cache metadata.

The current implemented tools are kiln-bench expert-pack and expert-pack-named, which write sidecar files from explicit validated mappings. They do not implement the proposed kiln convert KUMF workflow or train a prerouter.

## Open questions

- Is expert-contiguous sidecar layout sufficient for each supported model and I/O backend, or is block-level layout needed?
- How should fused expert tensors be mapped safely across architectures and quantization layouts?
- Should a future UMF be a GGUF-preserving sidecar bundle or a distinct container? The answer needs a compatibility test with stock llama.cpp.
- Which metadata belongs in a model artifact, and which must remain runtime-generated and hardware-specific?
- Phase 5.0 predictor feasibility and quality gates remain prerequisites for shipping trained prerouter artifacts.

# Phase 5.1 implementation: GGUF-preserving expert sidecar (November 2026)

**Compatibility correction:** Physically removing expert tensor bytes from a GGUF while claiming that stock llama.cpp can still execute the unchanged model is **not a proven GGUF-compatible operation**. The Phase 5.1 implementation therefore does not silently relocate or strip tensors inside the original GGUF. It writes **separate expert pack files** alongside an intact source model, preserving stock GGUF reading behavior.

The new `kiln-bench expert-pack` tool accepts a source file and a validated `ExpertTable` JSON mapping exact source byte ranges to `(layer, expert_id)` keys. It copies only those ranges, in bounded 64 KiB chunks, into a separate `experts.bin`, aligns each expert to a 4 KiB offset and writes:

1. `experts.index.json`: an existing-API-compatible `ExpertTable`, usable with `ExpertBlobReader::open_indexed`.
2. `experts.crc32.json`: versioned per-expert CRC32 integrity metadata with source size and alignment.
3. `experts.bin`: packed expert data, with zero padding between experts.

The original source is opened read-only; destination paths use create-new semantics (no replacement of existing files). Source expert ranges must pass version, sorted-key, non-overlap and source-file bounds validation before creating any output.

```sh
kiln-bench expert-pack \
  --source-file /models/original.gguf \
  --source-index /models/expert-source-ranges.json \
  --blob-file /models/experts.bin \
  --index-file /models/experts.index.json \
  --integrity-file /models/experts.crc32.json --json

kiln-bench expert-pack-verify \
  --blob-file /models/experts.bin \
  --index-file /models/experts.index.json \
  --integrity-file /models/experts.crc32.json --json
```

**Critical limitations:** This does **not** yet discover exact expert tensor ranges automatically from all GGUF/MoE variants. The source index must be produced by a trustworthy model-aware extractor or manually validated, not guessed. CRC32 detects accidental corruption but is **not** a cryptographic signature or authenticity guarantee. The format contains no trained prerouter, executable KILN-only tensor relocation instructions, or inference scheduler. No claims of running a stripped GGUF through stock llama.cpp or Phase 5 end-to-end MoE throughput are made.

Hosted tests verify exact source/expert bytes after pack-roundtrip, 4 KiB offsets and zero-filled padding, CRC corruption rejection, no overwrite of pre-existing files, and fail-before-writing for bad ranges. Later named-component work added explicit GGUF tensor maps plus TQ1_0/Q4_K size and roundtrip checks. Remaining Phase 5.1 gates are a complete map for the pinned target MoE, fused-tensor-aware extraction where required, and full model inference parity after sidecar materialization. The original Dell/streaming performance gates remain open.

# Phase 5.1 — Named GGUF tensor components to verified expert sidecars

This implements the **model-aware, non-destructive** part of the UMF expert-container requirement. It consumes an explicit list of original GGUF tensor names and their intended layer/expert IDs, validates their real on-disk offsets, conservatively checks the quantized byte extent and writes component-contiguous expert groups. It does **not** remove tensor bytes from the original GGUF.

## Source selection file

A deliberately explicit, reviewable JSON input avoids unsafe guesses about fused MoE tensor shapes. The tensor names must exist in the original GGUF. The example names below are illustrative only; use your model's actual GGUF tensor inventory.

```json
[
  {"layer":0,"expert_id":0,"tensor_name":"blk.0.ffn_down.0.weight"},
  {"layer":0,"expert_id":0,"tensor_name":"blk.0.ffn_gate.0.weight"},
  {"layer":0,"expert_id":0,"tensor_name":"blk.0.ffn_up.0.weight"},
  {"layer":0,"expert_id":1,"tensor_name":"blk.0.ffn_down.1.weight"}
]
```

A model with **fused expert tensor matrices** (e.g., one tensor containing all experts across a tensor dimension) is **not supported by this mapping**. Decomposing its component storage requires quantization-layout-aware slicing; treating a fused tensor as one expert would silently corrupt inference. Do not use this API to claim general model conversion.

## Invocation

```sh
kiln-bench expert-pack-named \
  --source-gguf /models/original.gguf \
  --selections-file /models/selected-tensors.json \
  --blob-file /models/experts.bin \
  --index-file /models/experts.index.json \
  --integrity-file /models/experts.crc32.json \
  --components-file /models/experts.components.json --json
```

The writer creates four files only if they do not already exist: (1) aligned `experts.bin` expert groups, (2) a validated `ExpertTable` index usable by `kiln-io` readers, (3) a per-expert CRC32 manifest and (4) a named tensor-component offset map. Each component retains its original raw byte representation (including TQ1.0 and supported K-block quantizers) with no re-quantization or lossy transformations. Copying is chunked, not a full-model duplicate in memory.

The runtime can use the expert index to locate whole expert groups, and the additional component map to reconstruct exact tensor byte boundaries. **Runtime loading of those tensor components into llama.cpp execution is not yet implemented.**

## Safety and validation

- Inputs are validated against actual parsed GGUF tensor names, array shapes, types and source-file bounds.
- Unsupported or size-overflowing tensor types are rejected rather than guessed.
- Duplicate tensor names, intersecting byte ranges and malformed component sequences are rejected.
- Components are sorted within each expert group, group offsets are aligned to 4 KiB and gaps are zero-filled.
- Output files use create-new semantics; invalid source mappings fail before any artifact creation.
- The verification function checks every expert's CRC32 and requires exact contiguous component coverage of each expert range.
- CRC32 is **accidental-corruption detection**, not cryptographic provenance or trust of the selection file. Validate model and mapping hashes separately for production.
- This is a sidecar alongside an unchanged, stock-loadable GGUF, not a claim that stripped expert weights are portable to stock llama.cpp.

Test coverage exercises exact TQ1.0 / Q4_K source-size calculations, integer-overflow and bounds failures, multiple components per expert, byte-perfect reassembly, zero padding, tampered component-map rejection and overlapping sources. Continue Phase 5 with actual fused-tensor-aware extraction, format parity, then streamed llama.cpp expert dispatch and DDR/NVMe metrics.
