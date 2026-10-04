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


## [Unreleased] — Phase 1: transformer works, KV cache in

### Added

- **KV cache** in `crates/kiln-core/src/kv_cache.rs`:
  - `LayerKv` per transformer layer, K and V stored post-bias post-RoPE.
  - `KvCache` owning one `LayerKv` per layer, plus `cached_tokens`.
  - `can_append`, `append_tokens`, `k_row`, `v_row` accessors.
- **docs/kv-cache-design.md** — full design, algorithm, tests, anti-patterns.
- **docs/avx2-matmul-design.md** — next unit of work.

### Changed

- **`Transformer::apply_rope`** now takes a `pos_offset` so cached K is
  not re-rotated. Absolute positions are required for correctness.
- **`Transformer::forward_tokens`** now takes `&mut self`, reuses the
  cache when the caller's token sequence starts with the cached prefix,
  resets when it does not, and processes only the new positions.
- **`kiln chat`** now streams tokens as they are produced and prints a
  `[Xs, Y tok/s, N tokens]` line at the end of each response. Replaces
  the previous behavior of buffering the entire reply before printing.

### Fixed

- **Q4_K byte layout** in `cpp/kernels/q4k.c`. The 128 weight bytes were
  being decoded as 8 chunks of 16 bytes with one scale each. The ggml
  layout is 4 groups of 32 bytes; within each group, the low nibbles and
  high nibbles map to different output regions with different scales.
  Every Q4_K linear layer was reading a permuted weight matrix. Top-1
  for "The capital of France is" went from token 99222 ("太", garbage)
  to 12095 (" Paris", correct).
- **Streaming UTF-8 display.** BPE space markers (`Ġ`) and newline
  markers (`Ċ`) were being printed raw. Now converted to `" "` and `"\n"`
  in the stream closure.

### Verified

- **Q6_K dequant** checked line by line against ggml's
  `dequantize_row_q6_K`. Already correct. No fix needed.
- **Chat output is bit-identical** to the pre-cache path with the same
  prompt, seed, and max tokens. Cache correctness is anchored.
- **First honest Tier 0 numbers.** Dell Latitude 7490, i7-8650U,
  16 GB DDR4-2400, no GPU, Qwen2.5-1.5B Q4_K_M (986 MB):
  - 5-token prompt forward pass: 9.32 s
  - 20-token reply, cached: 48.31 s
  - Throughput: 0.41 tok/s
  - Top-1 for "The capital of France is": " Paris" (correct)

### Known Issues

- No chat template. Qwen2.5 expects ChatML. Quality is off without it.
- KV cache overflow panics instead of returning an error. Needs a new
  error channel through the `Forward` trait.
- No differential test against ggml reference blocks. This is the
  missing anchor that would have caught the Q4_K bug.
- `clippy` is not installed. No lint gate.
- `monitor_sample` re-parses `/proc/cpuinfo` fully. 99–185 ms per call.
- `throughput_fraction` baseline is fragile.

## [Unreleased] — Phase 1.5: dispatch seam (Gate PASSED)

### Added

- **`crates/kiln-core/src/dispatch.rs`** — `Operation`, `Backend`,
  `ExecContext`, `Scratch`, `Dispatcher`, `DispatchError`. Rule
  KILN-E36 in code.
- **`crates/kiln-core/src/backends/cpu_scalar.rs`** — the single
  backend registered in Phase 1.5.
- **`docs/dispatch-integration-design.md`** — Phase 1.5 design.

### Changed

- **`Transformer`** owns a `Dispatcher`. One backend registered:
  `cpu_scalar`.
- **`dot_packed`** builds an `Operation::Matmul`, an `ExecContext`,
  and dispatches. No kernel is called directly from the transformer.
- **`ExecContext`** carries `weight_bytes: &[u8]` and `quant:
  QuantKind` directly. No HashMap lookup, no clone. This is the fix
  for the 1.55x regression seen mid-session.

### Verified

- **Byte-identical output.** `hi there` → `Hello! How can I help you
  today?` at greedy, seed 42, max 9 tokens. Same as c75225d.
- **0.25% overhead.** 58.98s vs 58.83s for the 9-token reply. Gate
  is within 5%.
- **Zero direct kernel calls in `transformer.rs`** outside the
  `#[allow(dead_code)]` legacy function.

### Gate 1.5: PASSED

Correctness: byte-identical.
Performance: within 0.25%.
Architecture: every operation goes through the dispatcher.

### Next

Phase 2 — ternary in the shipping path. Register
`cpu_ternary_tq1_0` as a second backend. The kernel exists and is
tested. The dispatcher picks it when weights are TQ1.0. No change to
`transformer.rs`. Target: 2x faster than Q4_K on the same model.

## [Unreleased] — Roadmap v6.0, Phase 2 target locked

### Changed

- **Roadmap rewritten as v6.0.** Phase 2 is now a measured target:
  beat Ollama on the dense 1.5B path. The gap is 127x (Ollama 464 ms,
  KILN 58,980 ms on the same model and prompt).
- **Multithreading moved into Phase 2.** It was in Phase 3 in v5.0.
  That was a sequencing error; it is the largest single win available.
- **Ternary moved to Phase 3.** The literature on this exact
  hardware (Qiita, Ternary-Bonsai-8B on i7-8650U) reports 0.7 tok/s
  on an 8B ternary model. On DDR4-2400 the bottleneck is memory
  bandwidth, not quantization format. Ternary is a size optimization
  on Tier 0, a speed optimization on AVX-512 and GPU.
- **T-MAC removed from Failed Experiments.** It was evaluated in the
  wrong integration point (per-row LUT, split dequant). It is now
  "Not yet fairly tested" and scheduled for retest in Phase 3.3.
- **"Q4_K is memory-bound, not compute-bound" corrected.** A recent
  Rust port of llama.cpp's kernel measured 9.3 GiB/s scalar vs 17.7
  GiB/s SIMD on the same data. If memory-bound they would be equal.
  It is compute/issue-bound. Better SIMD does help.

### Added

- **Rule KILN-E39** — a phase is self-contained. Every phase has
  Prerequisites, Deliverables, Success criterion, Out of scope, and
  Exit test.
- **Rule KILN-E40** — no commit without a measured gain.

### Status

- Phase 1: correctness reference. Complete.
- Phase 1.5: dispatch seam. Complete.
- Phase 2: beat Ollama. In progress. First unit is 2.1 multithreading.
- Phase 3: ternary. Not started.
- Phase 4: MoE streaming. Not started. This is the mission phase.

## [Unreleased] — Roadmap v7.0: fork llama.cpp

### The decision

**KILN is a layer on llama.cpp, not a from-scratch runtime.**

Measured today: Ollama 464 ms, KILN from-scratch 58,980 ms. 127x.
llama.cpp is MIT. Ollama wraps it. Every technique Ollama uses is
readable source. The gap is deleted by using the base, not solved.

### Added

- **Rule KILN-E41** — search before you build. Before writing any
  component, search for an existing implementation. If one exists
  and is licensed permissively, use it. The exception is KILN's
  differentiators: dispatcher, prerouter, UMF format, MoE streaming,
  FlashMoE, diffusion backend, adaptive management.

- **`docs/`** — the roadmap v7.0 lives at
  `~/KILN-MASTER-ROADMAP-AND-RULES-SET.md` and in the project folder.
  506 lines.

### Changed

- **Phase 2 is now "Fork and Wire."** Add llama.cpp as
  `vendor/llama.cpp`, pinned to a tagged release. Create
  `crates/kiln-runtime` wrapping its C API. Wire `kiln debug` to
  call it. Gate: ` Paris` top-1 in under 100 ms.

- **Phase 3** is dispatcher wired into the fork.
- **Phase 4** is ternary in the fork, using llama.cpp's BitNet path.
- **Phase 5** is MoE expert streaming. The mission phase.
- **Phase 6** is diffusion.
- **Phase 7** is hardware abstraction, using llama.cpp's backends.

### Deprecated

The from-scratch runtime. `kiln-kernels` and its C sources,
`transformer.rs`, `transformer_weights.rs`, `transformer_config.rs`,
`tokenizer.rs`, `chat.rs`, `kv_cache.rs`. They stay in the tree.
They are the reference for KILN-specific tests. They are no longer
developed.

### Kept from KILN

- `dispatch.rs` and the `Backend` trait (Rule KILN-E36)
- `QuantKind::row_bytes` and `QuantKind::detect`
- `kiln convert` (GGUF writer, TQ1.0 converter)
- The rules and the roadmap

## [Released]

None yet.
