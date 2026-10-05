# Unified Model Format (UMF) — Design

## Authority

Phase 5.1 of the KILN Master File. The UMF is the container that
makes MoE expert streaming possible. It is a KILN-specific format
layered on top of GGUF.

## Why a new format

GGUF stores tensors contiguously in a single data section. That is
correct for resident inference. It is wrong for streaming:

- Expert weights are interleaved with attention and norm weights.
  Every expert page-in reads a section of the file that includes
  bytes we do not need.
- There is no place to store the prerouter weights, the expert
  affinity graph, or the cache policy.
- There is no per-expert quantization marker.

The UMF fixes all three.

## The container

A UMF file is a GGUF file plus a KILN extension section. It is
readable by stock llama.cpp (the GGUF part), and by KILN (the whole
file).

Layout, from the start of the file:

1. GGUF header, metadata, tensor table. Standard, unchanged.
2. Tensor data section. Standard GGUF layout for attention, norms,
   embeddings, output.
3. KILN extension header. Magic KUMF, version 1, 32 bytes.
4. Expert manifest. One entry per expert per layer, 64 bytes each.
5. Expert data section. Contiguous per expert.
6. Prerouter weights. Per-layer predictor.
7. Cache policy metadata. FlashMoE parameters.
8. Thermal profile. Per-tier tuning.

## Thermal profile

Per-tier tuning hints:

- tier: u8, 0 to 4
- sustained_throughput_fraction: f32
- throttle_temp_c: u8
- recommended_n_threads: u8

Used by the runtime to preemptively adjust the DAG when the thermal
history predicts throttling. See Pillar 6.

## Compatibility

- Stock llama.cpp reads the GGUF portion and ignores the KILN
  extension. A UMF file loads and runs with no streaming.
- KILN reads both. When the extension is present, the streaming
  backend can use it. When it is absent, KILN falls back to
  llama.cpp built-in streaming on the plain GGUF.
- A plain GGUF with no extension is a valid input to KILN.

## Build process

kiln convert gains a UMF mode:

1. Read the source GGUF.
2. Extract expert tensors into the expert data section, one block
   per expert, contiguous.
3. Write the expert manifest.
4. Run the calibration pass and write the prerouter weights.
5. Write the cache policy.
6. Write the thermal profile.
7. Write the extension header.

Not in Phase 5.1: quantization conversion. The expert weights stay
in the source format.

## Open questions

- Expert-contiguous layout: is per-expert contiguity enough, or do
  we need per-expert-per-block contiguity for the io_uring path?
  Resolve during 5.3.
- Prerouter training feasibility: if the predictor cannot reach 80%
  accuracy at k=2 on Tier 0, the section is written but unused.
  Phase 5.0 decides.
- Hotness class assignment: static from calibration, or learned at
  runtime? Start static. Revisit if cache behavior suggests
  otherwise.
