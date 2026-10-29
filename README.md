# KILN

**A local LLM runtime designed to run models larger than the machine's physical memory.**

KILN is a per-layer runtime and dispatcher that coordinates compute and a unified memory hierarchy across CPU, integrated/discrete GPU, NPU, RAM, NVMe storage and optional VRAM. The product goal is one runtime contract, one model file where supported, and a local-first experience across ordinary laptops and larger systems.

> **Status and source of truth:** Phases 1, 1.5, 2.0–2.3, 3, 4 and 4.5 are recorded complete. Phase 3.5 (dense-path performance) is the immediate execution priority. Phase 5 (CPU-only MoE expert streaming) remains the core mission, with feasibility and end-to-end acceptance gates still pending. The full phase ledger, rules and research log live in the [KILN Master Roadmap and Rules Set](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md).

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
| 5 | CPU-only MoE expert streaming | In progress; end-to-end feasibility not yet demonstrated |
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
# Measure sequential vs random expert-block reads on a real model/expert blob
cargo run -p kiln-bench -- expert-io --file /path/to/experts.bin --block-bytes 1048576 --requests 256 --runs 5 --json
```

### Expert-block I/O feasibility probe

`kiln-bench expert-io` reads bounded blocks from a real file and reports sequential and deterministic random-access throughput plus p50/p95 request latency. It is an initial storage-I/O feasibility probe, not a model inference benchmark: it does not control the operating-system page cache, validate an expert index, or prove end-to-end MoE throughput. Run it against a representative `experts.bin` or other large model blob on the target laptop before using the numbers for design decisions.

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

## Current next steps

1. Reproduce the dense-path baseline and profile the actual hot path for Phase 3.5.
2. Test prompt-cache behavior, speculative decoding, thread counts and batch sizes as isolated experiments before combining them.
3. For Phase 5, measure realistic expert-block I/O, random-read latency, bytes fetched per token, cache usefulness, peak RAM and I/O/compute overlap before expanding the implementation.
4. Run cache/predictor/prefetch ablations, then validate the exact model end-to-end with sustained throughput, memory and quality measurements.
5. Keep the 70B-class ≥4 tok/s goal and Qwen3-30B-A3B ≥3 tok/s recorded targets clearly labelled as **targets until reproduced**.
6. Keep licensing review as a separate tracked task and update public performance claims only from reproducible artifacts.

## License

MIT. See [LICENSE](LICENSE).
