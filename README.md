# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model
file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad
in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

**Phase 5 preparation complete. Phase 3.5 — dense path speed run — is next.**

KILN forks llama.cpp, statically links it, routes every operation
through `RuntimeDispatcher`, runs natively ternary-trained models,
validates every model with `kiln doctor`, and watches every
generation for degeneration.

Measured on Tier 0, Qwen2.5-1.5B Q4_K, prompt `hi there`:

| | Ollama | KILN |
|---|---|---|
| Generation rate | 26.86 tok/s | **10.57 tok/s** |
| Prompt cache | 30/31 tokens | none |
| Total wall clock (warm) | 464 ms | 850 ms |

The from-scratch 127x gap is gone. The remaining gap is **2.5x on
generation rate**. Phase 3.5 closes it with four validated fixes:

1. **Append, do not reset.** Stop re-prefilling the ChatML wrapper
   every turn. Ollama caches 30/31 prompt tokens; we cache none.
   Expected 2-3x on turn 2+.
2. **Draftless n-gram speculative decoding.** `--spec-type ngram-mod`
   ships in llama.cpp. No draft model. 1.3-2x. Stacks with prompt
   cache.
3. **Thread tuning.** Test 2, 3, 4 threads on the real workload.
4. **Batch tuning.** Test `n_batch` 256-2048, `n_ubatch` 128-512.

**MoE streaming is built and waiting.** The `moe-stream` branch of
`github.com/andreipath26/kiln-llama` builds on Tier 0. All streaming
flags verified: `--moe-stream`, `--moe-stream-cache`,
`--moe-stream-io-threads`, `--moe-stream-direct`, `-ncmoe`. The model
test (Qwen3-30B-A3B) waits for data budget.

Phases 1, 1.5, 2.0-2.3, 3, 4, 4.5 complete. **Next is Phase 3.5:**
beat Ollama on the dense path. **Then Phase 5:** MoE expert
streaming. The mission.

The product claim: **KILN runs models larger than your machine's
memory.**

## What this is## What this is## What this is## What this is

KILN is a per-layer dispatcher over a unified memory hierarchy. It
schedules computations across CPU, iGPU, dGPU, NPU, RAM, NVMe SSD, and
optional VRAM as one managed pool. It is not a monolithic inference
engine.

## Design documents

- docs/architecture.md — six-layer runtime stack
- docs/core-design.md — Performance Monitor, Mode Selector, DAG Scheduler
- docs/kernels-design.md — TQ1.0 ternary format, C ABI, build process
- docs/fused-kernel-design.md — LUT-based matmul design and why it failed
- docs/pipeline-design.md — end-to-end pipeline
- docs/gguf-design.md — GGUF loader specification
- docs/tokenizer-design.md — BPE tokenizer
- docs/chat-loop-design.md — generation loop
- docs/transformer-design.md — forward pass
- docs/quantization-policy.md — Rule KILN-E35, supported formats
- docs/kv-cache-design.md — per-layer K/V cache
- docs/avx2-matmul-design.md — next unit of work

## Principles

1. The lowest-spec machine is the acceptance test.
2. One binary per platform, one model file, one API.
3. Per-layer dispatch, not per-machine modes.
4. Accuracy preserved.
5. Local by default.
6. Open-core. MIT or Apache-2.0.
7. Fail-safe.
8. The masses come first.

## Crates

- **kiln-hal** — Backend trait, Registry trait, core types
- **kiln-models** — UMF, synthetic format, GGUF loader, BPE tokenizer
- **kiln-mem** — Memory Hierarchy Manager
- **kiln-io** — I/O abstraction, IoSource trait, SyncIoSource
- **kiln-core** — Monitor, Selector, Scheduler, Pipeline, Chat loop
- **kiln-kernels** — TQ1.0 ternary pack, unpack, and fused matmul
- **kiln-api** — Ollama-compatible REST server on port 11435
- **kiln-cli** — Command-line interface with ten commands
- **kiln-bench** — Benchmark harness

## Commands

    cargo run -p kiln-cli -- chat --model <gguf> --seed 42 --max-tokens 50
    cargo run -p kiln-cli -- pipeline <file> --loader gguf
    cargo run -p kiln-cli -- info
    cargo run -p kiln-cli -- serve
    cargo run -p kiln-bench -- run

## Measured performance (Dell Latitude 7490, no GPU)

### Kernel layer

| Operation | Time per element | Speedup |
|---|---|---|
| Pack scalar | 10.79 ns | baseline |
| Pack AVX2 | 0.61 ns | **17.70x** |
| Unpack scalar | 2.94 ns | baseline |
| Unpack table | 2.49 ns | 1.12x |
| Fused matmul scalar | 2.75 ns | baseline |

### Pipeline (synthetic model, 4x1024)

| Stage | Time |
|---|---|
| Load | ~69,000 ns |
| Select | ~5,500 ns |
| Schedule | ~2,500 ns |
| Execute | ~13,400 ns |
| Total | ~91,000 ns |

Output is deterministic at 65060 across 13 runs.

### Documented kernel failures

| Approach | Speedup |
|---|---|
| LUT single-row matmul | 0.04x |
| LUT 8-row matmul | 0.34x |

The LUT approach does not work on Skylake. Do not retry without a
fundamentally different design.

## Repository layout

- crates/ — Rust orchestration, scheduler, kernels, and CLI
- cpp/ — C kernels
- backends/ — Hardware backends (Phase 4)
- docs/ — Design documents
- models/ — Curated model catalog (Phase 5)
- tests/ — Integration, acceptance, and performance tests
- scripts/ — Build and release scripts

## Development rules

Rule KILN-E34: after every session, update the documentation, append
the session log to the master roadmap, update this README, and push to
GitHub. The session is not finished until the push succeeds.

## License

MIT. See LICENSE.
