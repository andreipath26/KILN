# KILN

**A local LLM runtime designed to run models larger than the machine's physical memory.**

KILN is a per-layer runtime and dispatcher that coordinates compute and a unified memory hierarchy across CPU, integrated/discrete GPU, NPU, RAM, NVMe storage and optional VRAM. The product goal is one runtime contract, one model file where supported, and a local-first experience across ordinary laptops and larger systems. The Dell Latitude 7490 is the minimum acceptance machine, not the hardware boundary; CPU-only Phase 5 acceptance does not limit KILN's CPU, GPU, NPU, APU/iGPU or heterogeneous-memory scope.

> **Status and source of truth:** Phases 1, 1.5, 2.0–2.3, 3, 4 and 4.5 are recorded complete. Phase 3.5 (dense-path performance) is the immediate execution priority. Phase 5 (heterogeneous MoE expert streaming; CPU-only Tier 0 acceptance path) remains the core mission, with feasibility and end-to-end acceptance gates still pending. The full phase ledger, rules and research log live in the [KILN Master Roadmap and Rules Set](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md).

## Mission

The minimum acceptance platform is a Dell Latitude 7490 with Intel i7-8650U, 16 GB RAM, no discrete GPU and NVMe storage advertised at approximately 2 GB/s sequential throughput.

The core mission target is a 70B-class mixture-of-experts model with approximately 4B active parameters at **at least 4 tokens/s**, while keeping WikiText-2 accuracy within 1% of an FP16 reference under a pinned, reproducible protocol. This is a target, **not an achieved result**. Active parameter count alone does not prove feasibility; expert bytes read per token, random-read behavior, cache residency, compute and I/O overlap all matter.

## Verified phase status

| Phase | Scope | Status |
|---|---|---|
| 1, 1.5 | Foundations | Complete |
| 2.0–2.3 | llama.cpp integration and system profile | Complete |
| 3 | RuntimeDispatcher and chat/debug integration | Complete |
| 3.5 | Dense-path performance run | **Next execution priority** |
| 4 | Ternary/TQ1.0 path | Complete |
| 4.5 | Sanity and quality safeguards | Complete |
| 5 | Heterogeneous MoE expert streaming (CPU-only Tier 0 acceptance path) | In progress; end-to-end feasibility not yet demonstrated |
| 6 | Diffusion | Future research track |
| 7 | Broader hardware abstraction | Future |
| 8 | Productization | Future |

Relevant history includes runtime-dispatcher commits `d702aa6`, `140fc59`, TQ1.0 support `172bc92`, UTF-8 boundary handling `b27c869`, and quality/sanity checks `1033e45`. See the [commit history](https://github.com/andreipath26/KILN/commits/main) and the master roadmap for the fuller ledger.

## Runtime and crates

The Rust workspace contains these ten crates:

- **kiln-hal** — backend/registry traits and shared types
- **kiln-models** — model formats, UMF, GGUF loading and tokenizer support
- **kiln-mem** — memory hierarchy management
- **kiln-io** — I/O abstraction and data sources
- **kiln-core** — monitoring, selection, scheduling, pipeline and chat loop
- **kiln-kernels** — ternary packing/unpacking and matrix-kernel work
- **kiln-runtime** — runtime orchestration and dispatch integration
- **kiln-api** — Ollama-compatible REST API (default port 11435)
- **kiln-cli** — command-line interface
- **kiln-bench** — benchmark harness

Other top-level areas include `cpp/` (C kernels), `backends/` (hardware backends), `docs/` (design documents), `models/` (model catalog), `tests/` and `scripts/`.

## CLI

The CLI surface includes the following commands; exact flags depend on the command and build. Use the command's help output for the current syntax.

- `chat` — interactive generation
- `debug` — inspect/debug a forward pass
- `tokenize` — tokenize text
- `inspect` — inspect a model or artifact
- `info` — show runtime/build information
- `config` — inspect or validate configuration
- `weights` — work with model weights
- `convert` — model/format conversion
- `doctor` — model/runtime sanity and quality checks
- `pipeline` — run the pipeline against an input
- `serve` — start the API server
- `bench` / benchmark subcommands — run performance tests

Examples:

```sh
cargo run -p kiln-cli -- --help
cargo run -p kiln-cli -- chat --model <path-to-model.gguf> --seed 42 --max-tokens 50
cargo run -p kiln-cli -- debug --help
cargo run -p kiln-cli -- doctor --help
cargo run -p kiln-cli -- pipeline <input-file> --loader gguf
cargo run -p kiln-cli -- serve
cargo run -p kiln-bench -- run
```

## Design documents

The design record includes (availability and implementation status should be checked in the repository before relying on a document as an implementation contract):

- `docs/architecture.md`
- `docs/core-design.md`
- `docs/kernels-design.md`
- `docs/fused-kernel-design.md`
- `docs/pipeline-design.md`
- `docs/gguf-design.md`
- `docs/tokenizer-design.md`
- `docs/chat-loop-design.md`
- `docs/transformer-design.md`
- `docs/quantization-policy.md`
- `docs/kv-cache-design.md`
- `docs/avx2-matmul-design.md`
- `docs/dispatch-integration-design.md`
- `docs/runtime-loading-design.md`
- `docs/special-tokens-design.md`
- `docs/convert-design.md`
- `docs/runtime-dispatch-design.md`
- `docs/umf-design.md`
- `docs/moe-stream-build.md`
- `docs/edge0-study.md`

This list is a navigation aid, not a claim that every design is fully implemented. Check the file tree for the current inventory.

## Measured performance

Historical measurements on the Tier 0 Dell Latitude 7490 are shown below. They are **recorded results, not a guarantee of current-main performance**. Reproduce them with pinned model, build, backend, workload, cache state, power/thermal conditions and repeated runs before using them as release claims.

### Dense-path application results

| Measurement | Recorded result | Context |
|---|---:|---|
| `kiln debug` forward pass | 0.1497 s | System-profile result; a separate Phase 3 run recorded 0.1459 s best-case |
| `kiln chat` generation | 10.57 tok/s | Qwen2.5-1.5B Q4_K, prompt “hi there” |
| Ollama comparison | 26.86 tok/s | Same reported workload; prompt-cache behavior differs |
| KILN warm total wall clock | 850 ms | Historical comparison |
| Ollama warm total wall clock | 464 ms | Historical comparison |
| Prompt cache | KILN: none; Ollama: 30/31 tokens | Historical comparison |

The original from-scratch gap was reported as 127× and later reduced to approximately **2.5×** on generation rate. The 2.5× figure is the recorded remaining gap, not evidence that Phase 3.5 has already closed it. Prompt-cache behavior, generation settings and backend parity must be controlled in a fresh comparison.

### Kernel and synthetic-pipeline results

| Operation | Time | Recorded comparison |
|---|---:|---:|
| Scalar ternary pack | 10.79 ns/element | Baseline |
| AVX2 ternary pack | 0.61 ns/element | 17.70× faster than scalar |
| Scalar unpack | 2.94 ns/element | Baseline |
| Table unpack | 2.49 ns/element | 1.12× faster |
| Scalar fused matmul | 2.75 ns/element | Baseline measurement |
| Synthetic pipeline load | ~69,000 ns | 4×1024 synthetic model |
| Synthetic pipeline select | ~5,500 ns | 4×1024 synthetic model |
| Synthetic pipeline schedule | ~2,500 ns | 4×1024 synthetic model |
| Synthetic pipeline execute | ~13,400 ns | 4×1024 synthetic model |
| Synthetic pipeline total | ~91,000 ns | Output 65060 across 13 recorded runs |

### Ternary model result

Ternary-Bonsai-4B TQ1.0/group-64 was recorded as loading and sustaining coherent multi-turn conversation through `RuntimeDispatcher` on Tier 0. Historical throughput reports vary by run/build (approximately **0.72–0.85 tok/s**); preserve the exact benchmark artifact and environment when quoting a narrower figure. Group-128 compatibility was reported as unavailable in the tested mainline llama.cpp revision. This result does not establish the Phase 5 MoE mission target.

## Failed approaches and lessons

Negative results are part of the engineering record and must not be hidden:

| Approach | Recorded result | Conclusion |
|---|---|---|
| LUT single-row matmul | 0.04× | Slower than scalar; do not retry without a fundamentally different design |
| LUT 8-row matmul | 0.34× | Slower than scalar; same caveat |
| AVX2 fused matmul attempt | 0.13× | Regression versus scalar in the tested implementation |
| Split-AVX2 matmul attempt | 6.8× slower | Regression; retain as a failed experiment |
| T-MAC integration | Wrong integration point | Reassess where the technique belongs before another implementation attempt |

These measurements are implementation- and hardware-specific. Do not generalize them into proof that all LUT, AVX2 or T-MAC approaches fail. Preserve the experiment details and rerun only with a clear hypothesis and a reproducible benchmark.

## Development rules

The authoritative rules and their canonical history are maintained in the [master roadmap](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md). In brief:

- **KILN-E34:** every session updates documentation, appends the session log to the master roadmap, updates this README, and pushes successfully.
- **KILN-E35–E51:** preserve the complete canonical rule set in the master roadmap/rule registry. These cover quantization policy, roadmap authority, phase containment, evidence-based completion claims, licensing separation, prior-art review, pinned llama.cpp integration, hardware-tier inclusivity, measured performance changes, and preservation of repository history/worktree state.
- Do not call a phase complete because it builds or a component benchmark passes; use its end-to-end exit gate.
- Keep external research results separate from KILN measurements. Do not assume independent optimization gains compose.
- Before Git repair or branch cleanup, inspect status, logs, remotes, submodules, unreachable objects and backups; preserve uncommitted work.

## API implementation status

Set `KILN_MODEL_PATH` to a readable GGUF file before starting `kiln serve`. The server loads that model once at startup and routes `POST /api/generate` and `POST /api/chat` through the runtime dispatcher. Both routes support non-streaming JSON responses; the API-streaming branch adds newline-delimited JSON chunks when `stream: true`. The chat route currently builds a simple role-prefixed prompt rather than applying a model-specific chat template. Generation is greedy and only `options.num_predict` is accepted; unsupported options are rejected instead of silently ignored. `GET /api/tags` reports the configured model, but a multi-model registry, model pull lifecycle, and complete Ollama parity remain unimplemented.

## Small language models (SLMs): complementary track

KILN will evaluate resident, quantized **small language models** as a practical local-inference tier on ordinary hardware, while preserving the 70B-class MoE mission. Candidate 0.6B–4B models may be compared for structured tasks, tool-call quality and optional speculative drafting; larger candidates are comparisons, not shipping commitments.

This is a **research and benchmarking decision, not an implemented feature or demonstrated speedup**. The Phase 3.5 plan first reproduces the dense baseline, then measures SLM end-to-end latency, tokens/s, peak RAM, task correctness and no-speculation versus n-gram versus small-drafter results. Any later model-level routing stays above per-layer dispatch and preserves deterministic authorization. Expert-prefetch prediction remains gated on Phase 5 I/O feasibility. See the [authoritative SLM integration decision and gates](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md#slm-integration-decision--complementary-local-inference-tier-2026-11-06).

## November 2026 research audit — critical feasibility gates

The [master roadmap](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md#15-research-and-repository-audit--2026-11-06-new-evidence-no-phase-reordering) includes source-level assessments of MoE-CORE, APEX, BITCOS, Q-Strata, ZipMoE, SSD-LLaMA and related theoretical I/O work, plus a platform-stack survey spanning ggml backends, OpenVINO, Ryzen AI, Qualcomm, ONNX Runtime GenAI, MLX, IREE, MLC-LLM and ExecuTorch. These are research leads, not claimed KILN support.

**Two correctness/feasibility dependencies are tracked:** the byte-based logical tensor-admission fix has landed, but does not yet bound physical RSS, caller-owned buffers or OS page cache; the proposed UMF expert-relocation layout still needs proof of stock-GGUF compatibility. PR #15 added the expert-I/O and cache feasibility tools and a read-only Dell measurement runner; this research follow-up adds optional Linux process-attributed storage-read counters. Neither is a real-model benchmark. Actual SSD bytes/token, resident memory, format round trips and end-to-end model quality are mandatory before any streaming performance claim.

The literature suggests promising techniques—unequal layer cache allocation, confidence-aware *exact* prefetch, lossless expert compression and distribution-aware ternary layouts—but **none is a reproduced KILN speedup**. These are bounded experiments within existing phases, not a replacement for the 70B-class mission or an excuse to postpone Phase 3.5.

## Phase 3.5 measurement implementation (November 2026)

KILN now includes bounded prompt prefill, a thread/logical-batch/physical-microbatch benchmark matrix, explicit real decode-forward throughput measurements and Linux process RSS checkpoints. Run `kiln bench <model.gguf> --threads 4 --batch 512 --ubatch 128 --tokens 128 --runs 5 --json` on a pinned local GGUF to capture a controlled sample. See [benchmark rate contract](docs/phase35-decode-rate-contract.md), [native microbatch protocol](docs/phase35-native-microbatch.md) and [Dell matrix script](scripts/run_tier0_dense_matrix.sh).

**These implementations passed hosted CI; no new Dell benchmark or Ollama parity has been established.** The opt-in model differential test and the ≥27 tok/s target remain pending real target-machine validation.

### Phase 3.5 — optional interactive KV reuse

`kiln run <model.gguf> --keep-context` now preserves already executed conversation tokens across turns, instead of resetting the KV cache before every prompt. The opt-in mode uses generic dialogue delimiters, retains bounded token-position state and falls back to a fresh context when needed. Without this flag, `kiln run` continues to treat each prompt independently. The native wrapper also rejects batches that would overrun configured `n_ctx` before partially decoding them.

See [append-only KV behavior and its ignored GGUF differential test](docs/phase35-append-only-interactive-kv.md) and [context-capacity guard](docs/phase35-context-capacity-guard.md). These changes passed hosted CI, but actual Dell performance, model-specific chat correctness and the ≥27 tok/s target **remain unverified**.

## Speculative decoding and expert predictor experiments (November 2026)

The opt-in `kiln bench-spec` benchmark now compares bounded, draftless n-gram proposals verified in the actual llama.cpp target model with an ordinary target-greedy baseline. Rejected draft KV entries are rolled back, and the benchmark demands identical token IDs; see [Phase 3.5 speculative protocol](docs/phase35-speculative-ngram.md). **Hosted CI does not prove a target-machine speed improvement or run the ignored real-model differential test.**

Two Phase 5.0 offline feasibility commands are also implemented: `kiln-bench predictor-holdout` for a frozen chronological single-next-expert holdout and `kiln-bench router-holdout` for grouped per-token/per-layer MoE active expert sets. See [training pipeline](docs/training-pipeline-design.md) and [grouped router evaluation](docs/phase50-grouped-router-evaluation.md). Neither claims the 80% k=2 predictor gate is met on an actual checkpoint.

## Platform support and Phase 5 scope

KILN's architecture spans available CPU, GPU, NPU, APU/iGPU and memory tiers under one runtime contract. Phase 5's CPU-only route is the minimum Tier 0 acceptance path; it is not the product's deployment limit. Vendor stacks may require distinct graph formats or generated device artifacts, so every backend needs verified dispatch coverage, correctness parity, memory accounting and a safe CPU fallback. See the master roadmap's platform/runtime landscape for the project-by-project evaluation.

## Current next steps

1. Reproduce the dense-path baseline and profile the actual hot path for Phase 3.5.
2. Test prompt-cache behavior, speculative decoding, thread counts and batch sizes as isolated experiments before combining them.
3. For Phase 5, measure realistic expert-block I/O, random-read latency, bytes fetched per token, cache usefulness, peak RAM and I/O/compute overlap before expanding the implementation.
4. Run cache/predictor/prefetch ablations, then validate the exact model end-to-end with sustained throughput, memory and quality measurements.
5. Keep the 70B-class ≥4 tok/s goal and Qwen3-30B-A3B ≥3 tok/s recorded targets clearly labelled as **targets until reproduced**.
6. Keep licensing review as a separate tracked task and update public performance claims only from reproducible artifacts.

## License

MIT. See [LICENSE](LICENSE).
