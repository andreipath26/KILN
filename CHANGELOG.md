# Changelog

All notable changes to KILN are documented here.

The format is based on Keep a Changelog.

## [Unreleased]

### Phase 1 — Core Engine (2026-10-03)

The kernel layer, the pipeline, and the GGUF loader. KILN now reads real
GGUF files and runs inference.

#### Added

- **kiln-kernels crate** with TQ1.0 ternary packing.
  - Scalar pack and unpack in C11.
  - AVX2 SIMD pack path with runtime detection.
  - Table-based unpacker.
  - Scalar fused matmul.
  - Documented failed approaches: LUT single-row and LUT 8-row matmul.
- **GGUF loader** in kiln-models:
  - Header parser (magic, version, tensor count, metadata count).
  - Metadata parser for all nine GGUF value types including typed arrays.
  - Tensor table parser for F32, F16, Tq1_0, and unknown dtypes.
  - GgufFile with mmap and zero-copy tensor slicing.
  - Alignment handling via general.alignment metadata key.
- **Pipeline** in kiln-core:
  - run_once function that loads a model, selects a plan, schedules it,
    executes the fused matmul, and returns a timing report.
  - Loader enum with Synthetic and Gguf variants.
  - Real Selector and Scheduler wired through the pipeline.
- **CLI command** `kiln pipeline <path> [--loader synthetic|gguf] [--json]`.
- **Design documents**:
  - docs/kernels-design.md
  - docs/fused-kernel-design.md
  - docs/pipeline-design.md
  - docs/gguf-design.md
- **Rule KILN-E34**: end-of-session mandatory actions.

#### Measured on Dell Latitude 7490, no GPU

Kernel layer:
  pack scalar:          10.79 ns/element
  pack AVX2:             0.61 ns/element   (17.70x speedup)
  unpack scalar:         2.94 ns/element
  unpack table:          2.49 ns/element   (1.12x speedup)
  matmul scalar fused:   2.75 ns/element

Pipeline on a 4x1024 synthetic model:
  output: 65060, deterministic across 13 runs
  load:     ~69,000 ns
  select:    ~5,500 ns
  schedule:  ~2,500 ns
  execute:  ~13,400 ns
  total:    ~91,000 ns

Pipeline on a real 948-byte GGUF file (1 tensor, 4x1024, all trits +1):
  output:              -968
  nodes_executed:      4
  load:         ~78,000 ns
  select:        ~5,000 ns
  schedule:      ~2,000 ns
  execute:      ~13,900 ns
  total:       ~100,000 ns

#### Documented Failures

- LUT single-row matmul: 0.04x speedup.
- LUT 8-row matmul: 0.34x speedup.
- Table-based LUT build made it worse: cache pressure outweighed the
  division savings.
- Do not retry the LUT approach without a fundamentally different design.

#### Known Issues

- The fused scalar matmul at 2.75 ns/element gives 0.12 tok/s on a 3B
  active MoE. AT-2 requires 3 tok/s. The gap is 25x. The kernel alone
  cannot close it.
- load_time_nanos is 65-78 microseconds per pipeline run. Higher than
  the design estimate.
- monitor_sample re-parses /proc/cpuinfo fully. 99-185 ms per call.
- Item 8 (thermal envelope measurement) requires an AC-powered
  30-minute test before the Phase 2 gate.

### Phase 0B — Foundation (2026-10-03)

Fourteen commits. Eight crates created. Zero warnings.

#### Added
- Repository skeleton.
- docs/architecture.md, docs/core-design.md.
- phase-0a-findings.md.
- Eight crates: kiln-hal, kiln-models, kiln-mem, kiln-io, kiln-core,
  kiln-api, kiln-cli, kiln-bench.
- .gitignore, README.md, CHANGELOG.md, LICENSE.

### Phase 0A — Research Closure (2026-10-03)

Eight research items. Seven confirmed. One conditional.


### Phase 1 late — Tokenizer, Chat Loop, CLI (2026-10-03)

The tokenizer, chat loop, and CLI chat command. Phase 1 core engine is
complete.

#### Added

- **BPE tokenizer** in kiln-models:
  - BpeTokenizer::from_gguf reads tokenizer.ggml.tokens, merges,
    special token IDs, add_bos_token, add_eos_token
  - encode with iterative merge by rank
  - encode_with_special for BOS and EOS injection
  - decode with special token skipping
  - BpeTokenizer::from_parts public constructor for tests
  - TokenizerError with MissingMetadata, MalformedMerge, UnknownToken,
    InvalidType variants
  - Five tests: encode_decode_round_trip, encode_without_merges,
    encode_with_special_tokens, decode_skips_special_tokens,
    empty_input_produces_empty_output
- **Chat loop** in kiln-core:
  - Forward trait with forward() and vocab_size()
  - MockForward for tests, SyntheticForward for pipeline demos
  - SamplingStrategy enum: Greedy, TopK, TopP
  - Sampler with xorshift64 PRNG for determinism
  - ChatSession with generate and generate_streaming
  - ChatError enum
  - Four tests: deterministic_greedy, max_new_tokens_respected,
    eos_stops_generation, empty_logits_errors
- **Chat command** in kiln-cli:
  - kiln chat --model <gguf> --seed N --max-tokens N --strategy S
  - Loads tokenizer from GGUF metadata, builds SyntheticForward,
    loops reading stdin, streams decoded tokens
- **docs/chat-loop-design.md** anchoring the generation loop

#### What Phase 1 core engine includes

Nine crates. Forty-two commits. All compiling.

- TQ1.0 ternary kernel with AVX2 pack at 17.70x speedup
- Scalar fused matmul at 2.75 ns/element
- GGUF loader with header, metadata, tensor table, and mmap
- BPE tokenizer reading from GGUF metadata
- Chat loop with three sampling strategies and seeded PRNG
- Pipeline wired through the real Selector and Scheduler
- CLI with ten commands

#### What remains for a working assistant

- Real transformer forward pass (attention, layernorm, FFN)
- Real model weights in a GGUF with tokenizer metadata
- KV cache for efficient multi-token generation
- Thermal response in a live inference

Those are Phase 1 late and Phase 2.


## [Released]

None yet.
