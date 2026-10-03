# KILN Architecture

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Anchor document for every code change in Phase 0B and beyond.

## Non-goals

This document does not specify implementation. It specifies interfaces, data structures, and decision boundaries. Implementation lives in the crates.

## Overview

KILN is a per-layer dispatcher over a unified memory hierarchy. It is not a monolithic inference engine. It schedules computations across CPU, iGPU, dGPU, NPU, RAM, NVMe SSD, and optional VRAM as one managed pool.

The runtime has six layers. This document specifies the interfaces between them.

## Layer 1 — Backends

A backend is anything that can execute a computation. Each backend implements the Backend trait and registers itself with the Constraint-Aware Registry.

The Backend trait exposes backend id, device type, supported operations, supported precisions, memory model, available memory, device bandwidth, conversion cost, residency state, transfer latency, synchronization overhead, and thermal state.

DeviceType is one of: Cpu, Cuda, Rocm, Metal, Vulkan, WebGPU, OpenVINO, CoreML, QNN, WASM.

Precision is one of: FP32, FP16, BF16, INT8, INT4, TQ1_0, TQ2_0, TQ2P, TQ3P.

ResidencyState is one of: InL3, InRam, InVram, OnSsd, NotLoaded.

ThermalState is one of: Cold, Warm, Throttling, SeverelyThrottling.

## Layer 2 — Memory Hierarchy Manager

The manager owns the pool of storage tiers and exposes a single interface to the scheduler. It does not decide what to load. It executes load and evict decisions made by the scheduler.

The manager tracks four tiers:

- CPU L3 cache, tens of MB, latency approximately 1 nanosecond.
- RAM, 16 GB on Tier 0, latency approximately 100 nanoseconds.
- NVMe SSD, hundreds of GB, latency approximately 10 to 100 microseconds.
- Optional VRAM, depends on GPU, latency approximately 100 to 500 nanoseconds.

The manager exposes tier capacity, tier used, load, evict, residency, and rebalance.

RebalancePolicy is one of: MinimizeLatency, MinimizeWrite, MinimizePower, Custom.

## Layer 2.5 — Performance Monitor

The Performance Monitor sits between the Memory Manager and the Constraint-Aware Registry. It exposes the live performance envelope to the scheduler. The scheduler reads it before every significant decision.

The monitor samples every 500 milliseconds and records:

- CPU frequency per core.
- CPU package temperature.
- Per-backend throughput over a rolling 10-second window.
- RAM pressure and swap activity.
- SSD read and write throughput.
- Battery state if on battery power.

The monitor exposes current envelope, thermal history, predicted throttle, and a callback registration interface.

PerformanceEnvelope contains current throughput as a fraction of baseline, current thermal state, and current RAM headroom.

ThrottlePrediction is Some when the thermal history suggests throttling within the next 60 seconds.

## Layer 3 — Constraint-Aware Registry

The registry aggregates all registered backends and exposes a single query interface to the scheduler. It does not make decisions. It answers questions.

The registry exposes backends, backends_supporting, fastest_backend_for, conversion_path, and thermal_history.

The registry maintains a conversion cost matrix between every pair of backends and every pair of precisions. This matrix is computed once at startup and updated when the thermal state changes.

## Layer 4 — Execution Plan

The execution plan is a DAG. Each node is a computation annotated with three orthogonal properties. Each edge is a data transfer with a measured cost.

A node has: id, op, precision, placement, algorithm, estimated cost, estimated memory.

Algorithm is one of: Diffusion, TernaryAR, MoEStream, SpecDecode, Dense.

A node may use a different algorithm from the surrounding nodes. A model is not diffusion overall or autoregressive overall. Individual nodes may be either, selected by the Automatic Mode Selector at load time.

An edge has: from, to, tensor, transfer cost, transfer backend.

The plan itself has: nodes, edges, total estimated latency, total estimated memory, plan revision.

The plan is revocable. When the Performance Monitor reports a thermal event or throughput drop below a threshold, the scheduler rebuilds the plan and bumps the revision. All in-flight nodes complete under their original plan. All new nodes use the new plan.

## Layer 5 — Automatic Mode Selector

The selector runs at model load time and again whenever the Performance Monitor reports a significant event. It reads the model's UMF metadata and the current Performance Envelope, then produces an Execution Plan.

The selector logic:

1. Read model architecture from UMF metadata.
2. Read the current performance envelope from the monitor.
3. For each layer in the model, select an algorithm, precision, and placement.
4. Assemble the DAG.
5. Compute total estimated latency and memory.
6. If total estimated latency exceeds the interactive threshold, degrade the plan using the ladder.
7. If total estimated memory exceeds the RAM budget, degrade the plan using the ladder.

The degradation ladder is defined in Pillar 6 of the Master File. It is not duplicated here. The selector calls into the ladder implementation.

## Layer 6 — User API

The user API exposes three interfaces. They must all remain stable.

CLI. Commands: serve, run, pull, list, bench, plan, info.

Ollama-compatible REST. Endpoints: POST /api/generate, POST /api/chat, GET /api/tags, POST /api/pull, DELETE /api/delete, GET /api/version.

Python bindings. Embedded mode via kiln.Engine. Client mode via kiln.Client.

The API contract is defined in Section 6 of the Master File. It is not duplicated here.

## Unified Model Format

The Unified Model Format (UMF) is a container that wraps a GGUF model with KILN-specific metadata. It is not a new file format. It is a directory layout with a manifest.

Structure:

model-name.umf/
    manifest.json
    weights.gguf
    experts.bin
    experts.index
    kv-layout.bin
    thermal-profile.json
    ml-cache/layer-000.bin
    ml-cache/layer-001.bin

manifest.json contains model architecture, quantization type per tensor, expert routing table location and format, recommended execution strategy per hardware tier, required VRAM and RAM for each plan, source model URL and checksum, and license plus redistribution rights.

experts.bin is the expert-contiguous storage. Experts are laid out so that the whole-layer prefill path can load all experts in one shot with a small number of sequential reads.

experts.index maps expert IDs to byte ranges in experts.bin.

kv-layout.bin is the KVDRIVE layout: semantic-contiguity packing and layer-head partitioning. It tells the runtime where to place KV pages on SSD so that frequently co-attended keys and values are sequential.

thermal-profile.json contains two plans: the cold-start plan and the steady-state plan. The runtime loads the cold-start plan, monitors, and transitions to the steady-state plan when throttling is detected.

ml-cache/layer-NNN.bin is the FlashMoE ML cache for layer NNN. 113 KB per layer. Trained on routing traces for this specific model.

## Performance Monitor Integration

The monitor runs on a dedicated thread. It writes to a lock-free ring buffer that the scheduler reads. The scheduler does not block on the monitor.

When the monitor detects a thermal event, it fires the registered callbacks. The callbacks notify the scheduler. The scheduler reads the current envelope. If the envelope is below the degradation threshold, the scheduler revokes the current plan and produces a new one. The new plan uses the next rung of the degradation ladder. The scheduler records the transition in the change log.

When the monitor detects recovery, it waits for 60 seconds of sustained recovery before firing the callback. The scheduler restores the previous plan. The scheduler records the restoration.

## Open Questions

These are unresolved at draft time and must be resolved before the corresponding phase gate.

- How is the ML cache trained without a GPU? Phase 2.
- What is the exact format of the experts.index file? Phase 0B.
- How does the runtime detect SSM vs transformer at load time? Phase 1.
- How does the DAG handle loops for iterative algorithms like diffusion? Phase 1.
- What is the memory budget for the plan itself? Phase 0B.
- How does the runtime handle a model that exceeds the RAM budget even after full degradation? Phase 2.

## Version History

2026-10-03 — Initial draft. Skeleton and interface specifications only. No implementation details.
