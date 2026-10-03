# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model
file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad
in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

Phase 1 in progress. The kernel layer exists and is benchmarked. The
end-to-end pipeline runs. The GGUF loader works. KILN reads real GGUF
files and runs inference.

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
- docs/gguf-design.md — partial GGUF loader specification

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
- **kiln-models** — Unified Model Format, synthetic format, GGUF loader
- **kiln-mem** — Memory Hierarchy Manager
- **kiln-io** — I/O abstraction, IoSource trait, SyncIoSource
- **kiln-core** — Performance Monitor, Mode Selector, DAG Scheduler, Pipeline
- **kiln-kernels** — TQ1.0 ternary pack, unpack, and fused matmul
- **kiln-api** — Ollama-compatible REST server on port 11435
- **kiln-cli** — Command-line interface
- **kiln-bench** — Benchmark harness

## Commands

Run the pipeline on a model file:
    cargo run -p kiln-cli -- pipeline <file> --loader gguf

Run the pipeline on a synthetic model:
    cargo run -p kiln-cli -- pipeline <file> --loader synthetic

Print hardware profile:
    cargo run -p kiln-cli -- info

Serve the API:
    cargo run -p kiln-cli -- serve

Run the benchmark suite:
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

### Pipeline (real GGUF, 948-byte file)

| Stage | Time |
|---|---|
| Load | ~78,000 ns |
| Select | ~5,000 ns |
| Schedule | ~2,000 ns |
| Execute | ~13,900 ns |
| Total | ~100,000 ns |

### Documented kernel failures

| Approach | Speedup |
|---|---|
| LUT single-row matmul | 0.04x |
| LUT 8-row matmul | 0.34x |

The LUT approach does not work on Skylake. The LUT build cost dominates
even with 8-way reuse. The AVX2 gather is too slow to compensate. Do not
retry without a fundamentally different design.

## Default ports

KILN API: 11435
Ollama: 11434
Both can run side by side.

## Repository layout

- crates/ — Rust orchestration, scheduler, kernels, and CLI
- cpp/ — C kernels
- backends/ — Hardware backends (empty, Phase 4)
- docs/ — Design documents
- models/ — Curated model catalog (empty, Phase 5)
- tests/ — Integration, acceptance, and performance tests
- scripts/ — Build and release scripts

## Development rules

Rule KILN-E34: after every session, update the documentation, append
the session log to the master roadmap, update this README, and push to
GitHub. The session is not finished until the push succeeds.

## License

MIT. See LICENSE.
