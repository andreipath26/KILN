# KILN architecture

**Authority:** The [master roadmap](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) is the only source of truth for mission, engineering rules, phase order, status, and acceptance evidence. This page summarizes the architecture boundary; it does not declare future capabilities implemented.

## Product and runtime boundary

KILN is a local inference runtime intended to coordinate model execution across heterogeneous compute and memory tiers. The user-facing model/API path should remain stable while runtime policy selects supported operations and backends. KILN owns placement and dispatch policy; backend libraries provide the kernels and device execution they actually support.

The intended topology includes CPU, integrated/discrete GPU, NPU, RAM, accelerator memory, and NVMe where available. These are design targets, not a claim that every backend or placement path is currently available. Shared physical memory must be counted once; compute and bandwidth contention still need separate measurement. The Dell Latitude 7490 is the minimum acceptance machine, not the product boundary.

## Current implementation boundary

The repository contains a Rust workspace, a llama.cpp integration, GGUF discovery/loading, a TQ1_0 path, CLI and loopback API/UI, benchmark tools, and Phase 5 expert-sidecar/cache/I/O/byte-dispatch components. The README and roadmap describe the current status and its limitations.

In particular, Phase 5 components are not yet wired into the stock llama.cpp MoE inference graph. A GGUF parser accepting a file does not prove that the runtime supports its model architecture. Backend discovery/buildability does not prove KILN dispatch coverage. Do not present a target, research result, synthetic trace, component test, or CI pass as end-to-end acceptance.

## Design constraints

- Keep dispatch at operation/layer granularity rather than encoding fixed per-machine modes.
- Preserve exact model execution as the baseline; evaluate approximate routing or representations under separate quality gates.
- Keep local inference usable without cloud connectivity; explicit remote search/download actions remain distinct.
- Report measured values separately from estimates, targets, and external results.
- Fail closed when a required backend, model operation, artifact, or capability is missing.
- Treat release packaging and platform support as open until the clean-machine gates in [runtime loading](runtime-loading-design.md) pass.

For phase ordering, component history, hardware targets, and the subsystem map, see sections 3–5 and the phase ledger in the master roadmap.
