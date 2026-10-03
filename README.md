# KILN

The fastest local LLM runtime on Earth. One binary per platform. One model file. One API. Runs on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad in Sydney. No cloud. No GPU required. No compromise on accuracy.

## Status

Phase 0B — Foundation. Week 1. Repository skeleton and architecture document complete.

## What this is

KILN is a per-layer dispatcher over a unified memory hierarchy. It schedules computations across CPU, iGPU, dGPU, NPU, RAM, NVMe SSD, and optional VRAM as one managed pool. It is not a monolithic inference engine.

The architecture is defined in docs/architecture.md. The roadmap and rules are defined in KILN MASTER ROADMAP AND RULES SET.md. The Phase 0A research findings are in phase-0a-findings.md.

## Principles

1. The lowest-spec machine is the acceptance test.
2. One binary per platform, one model file, one API.
3. Per-layer dispatch, not per-machine modes.
4. Accuracy preserved.
5. Local by default.
6. Open-core. MIT or Apache-2.0.
7. Fail-safe.
8. The masses come first.

## Repository layout

- crates/ — Rust orchestration and scheduler
- cpp/ — C++ kernels and forked submodules
- backends/ — Hardware backends
- docs/ — Design documents and API contracts
- models/ — Curated model catalog
- tests/ — Integration, acceptance, and performance tests
- scripts/ — Build and release scripts

## License

MIT. See LICENSE.
