# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model
file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad
in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

Phase 1 in progress. The runtime loads real GGUF files, parses the
tokenizer metadata, runs a **real transformer forward pass** (RoPE,
GQA, RMSNorm, fused Q4_K / Q6_K / TQ1.0 matmul), keeps a per-layer
KV cache, and holds a streaming chat session.

First honest throughput on the acceptance-test hardware, Dell Latitude
7490, i7-8650U, 16 GB DDR4-2400, no GPU, Qwen2.5-1.5B-Instruct Q4_K_M
(986 MB): **0.41 tok/s on a 20-token reply, correct top-1 on a factual
prompt**. The floor is the scalar fused matmul. AVX2 is next.

Gate 0A: CONDITIONALLY PASSED. Seven of eight research items confirmed.
Item 8 (thermal envelope measurement) requires an AC-powered 30-minute
test before the Phase 2 gate.

## What this is

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
