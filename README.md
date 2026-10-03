# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model
file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad
in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

Phase 0B complete. All eight crates created. Phase 1 (Core Engine) begins
next.

Gate 0A: CONDITIONALLY PASSED. Seven of eight research items confirmed.
Item 8 (thermal envelope measurement) flagged for dedicated AC-powered
30-minute testing before the Phase 2 gate.

## What this is

KILN is a per-layer dispatcher over a unified memory hierarchy. It
schedules computations across CPU, iGPU, dGPU, NPU, RAM, NVMe SSD, and
optional VRAM as one managed pool. It is not a monolithic inference
engine.

The architecture is defined in docs/architecture.md. The core design is
in docs/core-design.md. The roadmap and rules are in KILN MASTER ROADMAP
AND RULES SET.md. The Phase 0A research findings are in
phase-0a-findings.md.

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
kiln-models   Unified Model Format schema
kiln-mem      Memory Hierarchy Manager
kiln-io       I/O abstraction, IoSource trait, SyncIoSource
kiln-core     Performance Monitor, Mode Selector, DAG Scheduler
kiln-api      Ollama-compatible REST server on port 11435
kiln-cli      Command-line interface with seven commands
kiln-bench    Benchmark harness

## Commands

Serve the API:       cargo run -p kiln-cli -- serve
Print hardware:      cargo run -p kiln-cli -- info
List commands:       cargo run -p kiln-cli -- --help
Run benchmarks:      cargo run -p kiln-bench -- run
Benchmark JSON:      cargo run -p kiln-bench -- run --json

## Default ports

KILN API:  11435
Ollama:    11434 (KILN does not bind to this port so that Ollama can
                  run side by side on the same machine)

## Repository layout

- crates/     Rust orchestration and scheduler
- cpp/        C++ kernels and forked submodules
- backends/   Hardware backends
- docs/       Design documents and API contracts
- models/     Curated model catalog
- tests/      Integration, acceptance, and performance tests
- scripts/    Build and release scripts

## License

MIT. See LICENSE.
