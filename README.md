# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model
file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad
in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

Phase 1 in progress. The end-to-end pipeline runs. Every crate is wired
together and invoked through one CLI command. The first ternary kernel
is compiled and benchmarked. Determinism is verified.

Gate 0A: CONDITIONALLY PASSED. Seven of eight research items confirmed.
Item 8 (thermal envelope measurement) requires an AC-powered 30-minute
test before the Phase 2 gate.

## What this is

KILN is a per-layer dispatcher over a unified memory hierarchy. It
schedules computations across CPU, iGPU, dGPU, NPU, RAM, NVMe SSD, and
optional VRAM as one managed pool. It is not a monolithic inference
engine.

The architecture is defined in docs/architecture.md. The core design is
in docs/core-design.md. The kernel design is in docs/kernels-design.md.
The fused matmul design is in docs/fused-kernel-design.md. The pipeline
design is in docs/pipeline-design.md.

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

kiln-hal      Backend trait, Registry trait, core types
kiln-models   Unified Model Format, synthetic model format
kiln-mem      Memory Hierarchy Manager
kiln-io       I/O abstraction, IoSource trait, SyncIoSource
kiln-core     Performance Monitor, Mode Selector, DAG Scheduler, Pipeline
kiln-kernels  TQ1.0 ternary pack, unpack, and fused matmul kernels
kiln-api      Ollama-compatible REST server on port 11435
kiln-cli      Command-line interface
kiln-bench    Benchmark harness

## Commands

Serve the API:       cargo run -p kiln-cli -- serve
Print hardware:      cargo run -p kiln-cli -- info
Run the pipeline:    cargo run -p kiln-cli -- pipeline <file>
List commands:       cargo run -p kiln-cli -- --help
Run benchmarks:      cargo run -p kiln-bench -- run
Benchmark JSON:      cargo run -p kiln-bench -- run --json

## Measured performance (Dell Latitude 7490, no GPU)

Kernel layer:
  pack scalar:          10.79 ns/element
  pack AVX2:             0.61 ns/element   (17.70x speedup)
  unpack scalar:         2.94 ns/element
  unpack table:          2.49 ns/element   (1.12x speedup)
  matmul scalar fused:   2.75 ns/element

Pipeline (4x1024 synthetic model):
  output is deterministic across 13 runs
  load:      ~69,000 ns
  select:    ~5,500 ns
  schedule:  ~2,500 ns
  execute:   ~13,400 ns
  total:     ~91,000 ns

Kernel approaches tried and documented as failed:
  LUT single-row matmul:  0.04x speedup
  LUT 8-row matmul:       0.34x speedup
  See docs/fused-kernel-design.md and the historical notes in the source.

## Default ports

KILN API:  11435
Ollama:    11434
Both can run side by side.

## Repository layout

- crates/     Rust orchestration, scheduler, kernels, and CLI
- cpp/        C kernels and forked submodules
- backends/   Hardware backends
- docs/       Design documents and API contracts
- models/     Curated model catalog
- tests/      Integration, acceptance, and performance tests
- scripts/    Build and release scripts

## License

MIT. See LICENSE.
