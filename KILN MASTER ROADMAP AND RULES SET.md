# KILN MASTER ROADMAP AND RULES SET

**Version:** 9.1 — current repository reconciliation  
**Last updated:** 2026-11-06  
**Status:** Active master roadmap  
**Repository:** https://github.com/andreipath26/KILN  
**Runtime fork:** https://github.com/andreipath26/kiln-llama

> **Authority and preservation rule:** This is the single authoritative KILN roadmap in the repository. Preserve the mission, rules, phase ordering, completed-work history, targets, and research leads. The research integrated here supplements and sharpens execution gates; it does not erase earlier ideas, invalidate phases, or turn external papers' results into KILN results. When a technique is not currently applicable, retain it in the research log with its hardware tier and reason. Update this document append-only where practical and record changes in the session log.

## 1. Mission and acceptance target

KILN is a local inference runtime intended to run models larger than the machine's physical memory by coordinating compute and a unified memory hierarchy. It is not a monolithic inference engine and is not limited to one hardware tier.

**Primary acceptance platform:** Dell Latitude 7490, Intel i7-8650U, 16 GB RAM, no discrete GPU, NVMe storage advertised at approximately 2 GB/s sequential throughput.

**Core mission target:** run a 70B-class mixture-of-experts (MoE) model with approximately 4B active parameters at **at least 4 tokens/s**, while keeping WikiText-2 accuracy within 1% of the FP16 reference under a pinned, reproducible evaluation protocol.

The 70B/4B target is a mission hypothesis and acceptance goal, not proof that active parameter count alone makes it feasible. Active parameter count estimates part of compute demand; it does not reveal expert bytes fetched, random-read latency, cache reuse, read amplification, synchronization cost, or how much I/O can overlap with compute.

**Product thesis:** “KILN runs models larger than your machine’s memory, at usable speed, on the hardware you already own.”

The Latitude 7490 is the minimum acceptance platform, not the product's mission boundary. Keep a common runtime contract and allow hardware-aware policies and kernels for laptops, iGPU/dGPU/NPU systems, servers, and other tiers.

## 2. Non-negotiable principles and engineering rules

1. **Lowest-spec acceptance first.** Higher-end hardware must not conceal failure on the target.
2. **One binary per platform, one model file, one API.** Preserve the product goal while using internal hardware-aware dispatch.
3. **Per-layer dispatch, not per-machine modes.** Route operations through the dispatcher and make placement decisions at the narrowest useful granularity.
4. **Accuracy preserved.** Every lossy representation or approximation needs a defined error budget and quality evaluation.
5. **Local by default.** Inference must not require cloud connectivity.
6. **Open-core and license discipline.** The repository currently declares MIT. Review dependencies, model licenses, and any proposed licensing change separately; do not assume legal conclusions.
7. **Fail-safe behavior.** Dispatch, memory-placement, or resource-discovery errors must not silently produce invalid results or corrupt state.
8. **The masses come first.** Prioritize useful performance on ordinary hardware, not only datacenter benchmarks.
9. **No commit without measured gain** for performance work. Keep a reproducible baseline and reject changes that show no benefit unless needed for correctness.
10. **Search before building.** Review prior art before inventing a subsystem; document what is borrowed conceptually, what is reimplemented, and why.
11. **Narrow llama.cpp seam.** Keep integration focused and stable; pin the exact upstream revision and maintain the andreipath26/kiln-llama fork/submodule deliberately.
12. **Phase-contained work.** Do not let a future phase silently expand the current phase. Record cross-phase research without changing phase order unless evidence and a logged decision justify it.
13. **All tiers matter.** Preserve relevant techniques even if the original work targets GPUs, datacenters, or another platform. Classify by technique, hardware tier, workload, phase, evidence quality, and reproduction status.
14. **Separate research from claims.** Distinguish published result, KILN hypothesis, implemented technique, and reproduced KILN result.
15. **No component metric substitutes for end-to-end performance.** Cache hit ratio, predictor accuracy, kernel speed, nominal bits/parameter, or a microbenchmark cannot substitute for sustained tokens/s, latency, memory use, and quality.
16. **Do not assume gains compose.** Test optimizations in isolation first; only then test combinations and re-profile the complete runtime.
17. **Preserve history and worktree state.** Before Git repair, branch cleanup, or large changes, inspect and preserve branches, uncommitted work, and backups. Never delete work merely because it looks obsolete.
18. **Append session log and push.** Rule KILN-E34: after each session, update documentation, append the session log to this roadmap, update README, and push to GitHub. The session is not complete until the push succeeds.
19. **Rule KILN-E35 — quantization policy.** Supported formats and quantization decisions are governed by docs/quantization-policy.md; keep it and this roadmap consistent.
20. **Roadmap is the single source of truth.** Update phase status only from verified repository state and reproducible evidence. README is a concise public summary, not a replacement.
21. **No unverified completion claims.** “Built,” “wired,” “tested,” “measured,” and “complete” mean different things. State what has actually been demonstrated.
22. **Licensing work is separate.** Complete licensing review as its own tracked task; do not mix licensing decisions into performance commits.
23. **External code is not automatically vendored.** Prefer extracting/reimplementing relevant techniques behind KILN interfaces; check license, dependency, maintenance, and architecture costs first.
24. **Pin experiments.** Record KILN commit, llama.cpp revision, build flags, compiler, model/tokenizer hashes, hardware, operating conditions, workload, and evaluation settings for every performance claim.

The project’s E36–E51 rule set also establishes operating constraints around dispatcher hot-path behavior, roadmap authority, phase-contained work, measured performance changes, prior-art review, narrow llama.cpp integration, pinned revisions, inclusive research, hardware tiers, and evidence-based handling of user-provided material. Preserve the exact canonical wording of those rules when editing the original rule registry; do not weaken them in implementation notes.

## 3. Architecture and repository map

KILN is a per-layer dispatcher over a unified memory hierarchy. It schedules work across CPU, integrated GPU, discrete GPU, NPU, RAM, NVMe SSD, and optional VRAM as available. Data placement and compute scheduling are coupled but should be profiled separately. The Latitude 7490's CPU-only configuration is the minimum acceptance path, not the supported-hardware boundary. Every hardware tier is a first-class target under the same runtime contract; device-specific backends, kernels, and memory policies must remain replaceable and fall back safely.

**Backend and memory contract.** Keep the user-facing model/API path stable while discovering backends through the llama.cpp/ggml registry and device-property interfaces where applicable. KILN owns placement policy and per-layer scheduling; backend libraries own their supported kernels and execution. Distinguish a backend being buildable from KILN actually dispatching work to it. Treat CPU, discrete GPU, and unified-memory APU/iGPU/NPU configurations as different topologies: shared physical memory must be counted once, while bandwidth contention and compute capacity are profiled separately. Some accelerator stacks require generated or compiled target artifacts; investigate derived, keyed caches without changing the one-source-model product goal until an explicit format design review resolves the trade-off.

- **kiln-hal:** backend and registry traits; shared core types.
- **kiln-models:** UMF, synthetic format, GGUF loader, tokenizer support.
- **kiln-mem:** memory hierarchy manager.
- **kiln-io:** I/O abstraction, IoSource, synchronous I/O source.
- **kiln-core:** performance monitor, mode selector, DAG scheduler, pipeline, chat loop.
- **kiln-kernels:** TQ1.0 ternary packing/unpacking and matmul kernels.
- **kiln-api:** Ollama-compatible REST API, default port 11435.
- **kiln-cli:** CLI and user-facing commands.
- **kiln-bench:** benchmark harness.
- **cpp/:** C kernels; **backends/:** hardware backends; **docs/:** design documents; **models/:** curated model catalog; **tests/:** integration, acceptance and performance tests; **scripts/:** build and release scripts.

Design documents referenced by README include docs/architecture.md, docs/core-design.md, docs/kernels-design.md, docs/fused-kernel-design.md, docs/pipeline-design.md, docs/gguf-design.md, docs/tokenizer-design.md, docs/chat-loop-design.md, docs/transformer-design.md, docs/quantization-policy.md, docs/kv-cache-design.md, and docs/avx2-matmul-design.md. Keep them aligned with actual implementation.

## 4. Current phase ledger

The latest supplied project state records these statuses. Reconfirm against Git, tests, and benchmarks before changing them.

| Phase | Scope | Status recorded | Notes / dependency |
|---|---|---|---|
| 1 | Core foundations | Complete | Preserve tests and history. |
| 1.5 | Foundation follow-up | Complete | Preserve tests and history. |
| 2.0–2.3 | llama.cpp fork/wiring and integration | Complete in supplied status | Keep fork/submodule revision pinned; verify Git state. |
| 3 | Runtime dispatcher | Complete | Protect dispatcher hot path. |
| 3.5 | Dense-path speed run | **In progress / next execution phase** | Controlled baseline, profiling, prompt cache, speculation, thread and batch settings. |
| 4 | Ternary weights / TQ1.0 path | Complete in supplied status | Verify effective storage and runtime representation. |
| 4.5 | Sanity and quality guarantees | Complete in supplied status | Keep quality checks for lossy representations. |
| 5 | MoE expert streaming across heterogeneous compute and tiered memory | **Core mission phase; full feasibility/end-to-end gates pending** | Merged PRs #41–44, #47, #49, #52, #53, #55 and #56 added GGUF sidecar, cache, guarded I/O, async reader, Linux io_uring and `cpu_moe_streamed` byte-operation dispatch components. The CPU-only Tier 0 route is the minimum acceptance slice. Actual llama.cpp MoE graph integration and real-model Dell acceptance remain open. |
| 6 | Diffusion | Future / not started | Separate research track; not a drop-in autoregressive optimization. |
| 7 | Hardware abstraction | Future / not started | Preserve common runtime contract and hardware-specific options. |
| 8 | Productization | Future / not started | Packaging, support, quality, performance and user-facing acceptance. |

Phase 5 remains the central mission, but Phase 3.5 is the immediate execution priority. Do not skip the dense-path baseline or storage feasibility gate because MoE code compiles.

### Phase 3.5 recorded baseline and target

The repository README records a Tier 0 Qwen2.5-1.5B Q4_K comparison on the Dell Latitude 7490 with prompt “hi there”:

| Measurement | Ollama | KILN |
|---|---:|---:|
| Generation rate | 26.86 tok/s | 10.57 tok/s |
| Prompt cache | 30/31 tokens | none |
| Warm total wall clock | 464 ms | 850 ms |

The original dense-path target was to meet or exceed a comparable Ollama baseline, previously expressed as 27 tok/s versus 26.86 tok/s. These are historical figures, not proof of current-head performance. Reproduce under a pinned protocol before using as current results.

Planned optimizations to test separately:
1. Append to prompt/cache state rather than resetting and re-prefilling the ChatML wrapper each turn.
2. Test draftless n-gram speculative decoding supported by the pinned llama.cpp revision. README mentions --spec-type ngram-mod; verify flag availability and semantics for the pinned commit.
3. Test thread counts 2, 3, and 4 on the real workload.
4. Tune n_batch (recorded range 256–2048) and n_ubatch (recorded range 128–512).
5. Measure --n-cpu-moe systematically where supported; pin exact flag semantics to the backend revision.

### Other recorded results

- AVX2 pack kernel: 0.61 ns versus scalar 10.79 ns per element, reported 17.70× speedup.
- Unpack table: 2.49 ns versus scalar 2.94 ns per element, reported 1.12× speedup.
- Fused matmul scalar: 2.75 ns per element.
- LUT single-row matmul: 0.04×; LUT 8-row matmul: 0.34×. Recorded conclusion: this LUT approach does not work on Skylake; do not retry without a fundamentally different design.
- Synthetic pipeline (4×1024): load ~69,000 ns; select ~5,500 ns; schedule ~2,500 ns; execute ~13,400 ns; total ~91,000 ns. Output was deterministic at 65060 across 13 runs.
- Edge0 prerouter A/B was recorded as 20.4 versus 19.9 tok/s with/without prerouter. This result applies only to that experiment.
- Phase 5 targets recorded earlier include predictor feasibility ≥80% at k=2, cache hit ≥85%, peak RAM below 15 GB, and Qwen3-30B-A3B at ≥3 tok/s. These remain targets until reproduced on defined hardware/workload. They are distinct from the top-level 70B-class ≥4 tok/s mission.

## 5. Phase 3.5 — Dense path protocol and exit gate

### Pin the experiment

Record KILN commit/worktree state; llama.cpp revision; compiler/build flags; model/tokenizer hashes; quantization; context and prompt; generation length; sampling/seed; threads; batch settings; OS/kernel; available memory; power/thermal state; cache condition; command; logs; run count; median and spread; output correctness.

Compare KILN and a pinned baseline on the same machine, model, prompt, settings and cache condition. Separate prompt-processing throughput from generation throughput. Warm up, repeat runs, report median and spread, and include sustained results after thermal stabilization. Ollama is a useful user-facing comparison, but is not automatically apples-to-apples unless backend and settings are established.

### Profile before optimizing

Measure time/resource use in dispatcher, sampler, decode, dequantization, matrix kernels, synchronization, KV-cache access, memory copies and I/O. Identify the largest current bottleneck before choosing the next change. Re-profile after a successful optimization because the bottleneck may move.

### Speculative decoding matrix

Compare one at a time: no speculation; supported draftless n-gram variants in the pinned llama.cpp revision; a small draft model if CPU and memory costs permit; other supported methods only where justified by the model and target.

Record accepted draft tokens per verification pass, verification overhead, output equivalence under the specified decoding setup, peak memory, prompt throughput, generation throughput and end-to-end latency. Acceptance rate alone is not a performance result.

### Thread and batch matrix

Run repeatable tests for 2/3/4 threads and the planned batch/micro-batch range. Keep prompt-cache, speculative decoding, threading, batch and --n-cpu-moe experiments separate before testing combinations.

**Phase 3.5 exit gate:** reproducible dense baseline; profile identifying largest costs; controlled comparisons to pinned baseline; measured evidence for each retained optimization. Update README figures only after rerunning the benchmark.

## 6. Phase 4 / 4.5 — Quantization, representation and quality

### TQ1.0 ternary weights

Recorded project state says TQ1.0 is already a 1.6-bit representation and no code change is required solely to make that claim. Verify exact encoding and distinguish nominal code width from effective bytes/parameter. Effective size must account for scales, metadata, padding, residuals and auxiliary buffers where present.

Do not publish “1.6-bit” as equivalent to effective 1.6 bits/parameter unless the complete representation supports it. Storage reduction is not automatically a speed improvement; measure decode/conversion overhead and target-CPU kernel performance.

### Quality gate

WikiText-2 within 1% of FP16 remains one quality gate. Define whether “1%” means relative or absolute change, pin model/tokenizer/evaluation protocol, and retain a stable reference. Add task-appropriate tests for instruction following, reasoning, coding, multilingual behavior and long-context retrieval as appropriate.

Every approximation, quantization, surrogate, pruning method or reduced-precision cache needs a quality test and explicit error budget. Keep published paper results distinct from KILN results.

### Ternary K/V and RSD blocks

Keep ternary K/V and the recorded RSD-block prototype as a separate Phase 4 research item. Measure effective bytes per context token, attention latency, quality/perplexity, long-context retrieval, numerical stability and memory peak. Ternary weight results do not establish ternary KV quality; distributions, access patterns and error propagation differ.

### QTEA and residual compensation

Research lead: QTEA, “Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization,” arXiv:2609.00224: https://arxiv.org/abs/2609.00224

Evaluate against existing TQ1.0 using the same model, tokenizer, hardware and evaluation suite. Treat reported bit count, quality and speed as paper results until reproduced. Target-CPU kernel performance and complete representation size matter as much as nominal precision.

## 7. Phase 5 — MoE expert streaming across heterogeneous compute and tiered memory

Phase 5 aims to stream MoE experts through the memory hierarchy while retaining a useful hot cache, predicting expert demand, and overlapping data movement with computation. The CPU-only Tier 0 path is the minimum acceptance route, not the product's hardware scope; the same per-layer runtime must use available CPU, GPU, NPU, APU/iGPU, and memory tiers where supported. Verify implementation and model-specific flags against the pinned andreipath26/kiln-llama revision. Preserve exact router execution as the primary correctness path; prediction schedules prefetch unless an explicitly separate approximation mode passes its own quality gate.

**Janas comparison.** Janas (https://github.com/prabanta-dev/janas) is evidence that a practical RAM cache for oversized MoE weights can work as a baseline. The reviewed project does not establish a trained next-layer prerouter or FlashMoE-style ML cache replacement. KILN Phase 5.2's graph-based multi-layer prerouter plus Commit Guard and Phase 5.3's asynchronous I/O plus learned replacement are proposed improvements on top of that baseline, not capabilities to attribute to Janas. Treat all performance/quality benefits as hypotheses until KILN reproduces them.

### Current Phase 5 implementation on main (reconciled 2026-11-06)

- **Phase 5.1:** PRs #41–44 add explicit GGUF tensor-to-expert component maps, a 4 KiB-aligned separate sidecar pack, CRC32 verification, and component readers. The original GGUF remains unchanged. Mapping is explicit; fused expert tensors are not generally decomposed automatically.
- **Phase 5.2/5.3:** PRs #15, #31, #47, #49, #52, #53 and #55 add bounded expert-file reads, byte-bounded cache/admission, authoritative-router Commit Guard behavior, bounded asynchronous prefetch, and a Linux io_uring reader with buffered fallback.
- **Phase 5.4:** PR #56 registers `MoEStreamBackend` (`cpu_moe_streamed`) for token-scoped expert-byte operations. A real-file test covers predictions, misses, router order, and stale epochs.
- **Integration boundary:** These are real code and test milestones, not end-to-end streamed inference. The stock llama.cpp model graph does not call the new expert operations or materialize sidecar tensor components into its compute path. Dell real-model speed, physical RSS, SSD traffic per token, quality, and Phase 5 exit criteria remain open.



### Existing design targets (not universal guarantees)

Recorded Phase 5 items: UMF container and bit-slicing; multi-layer prerouter plus commit guard; asynchronous I/O investigation; FlashMoE/RF-cache research; registration of cpu_moe_streamed; predictor feasibility ≥80% at k=2; cache hit ≥85%; peak RAM below 15 GB; Qwen3-30B-A3B at ≥3 tok/s. The `cpu_moe_streamed` name identifies the required Tier 0 path, not a CPU-only product boundary. These differ from the top-level 70B-class ≥4 tok/s mission. State exactly which model, hardware tier and target each result addresses.

### Gate 5A — Measure storage/RAM/compute ceiling first

Before expanding the complete streaming runtime, benchmark representative expert blocks in the intended on-disk layout/access pattern. Measure sequential and random reads at realistic block sizes; cold-cache versus warm-page-cache behavior; page faults; read amplification; queue depth and latency distribution; useful versus wasted bytes; RAM bandwidth and CPU kernel throughput; storage bytes and total memory bytes per generated token; peak resident memory; thermal stability; synchronization/scheduling cost.

Do not equate advertised NVMe sequential throughput with effective random/block-read throughput. First-order bound:

**storage-limited tokens/s ≤ usable storage bytes/s ÷ storage bytes read per generated token**

This only bounds the storage-served portion; it excludes compute, KV-cache, scheduling and other overhead. Measure the fraction served from cache and the fraction of I/O hidden by overlap. If the measured bound misses the target, assess cache residency, locality, representation size, alternative models and target assumptions before increasing implementation scope. A simplified bandwidth calculation alone neither proves feasibility nor infeasibility.

### Synthetic expert-cache experiment — exploratory, not a hardware result

**Recorded 2026-10-29:** a synthetic trace of 100,000 expert accesses across 128 experts produced these LRU hit rates:

| Cache capacity | Observed synthetic LRU hit rate |
|---:|---:|
| 4 experts | 18.76% |
| 8 experts | 36.84% |
| 16 experts | 68.50% |
| 32 experts | 88.70% |

These figures demonstrate only that cache capacity can materially affect hit rate under the chosen synthetic access pattern. They are **simulation results, not a measurement on a Dell/Latitude system, not a real-model trace, and not evidence of tokens/s or Phase 5 feasibility**. The trace generator/distribution, seed, exact cache semantics and run command must be recorded before treating the result as a reproducible baseline.

**Required follow-up benchmark work (not yet completed):**
1. Implement a byte-bounded cache; expert-count capacity alone is not representative when experts differ in size.
2. Compare LRU, LFU and an adaptive policy on the same pinned traces and memory budget.
3. Add deterministic trace replay and record the trace schema, generator/seed or source hash, cache configuration and results.
4. Measure hit/miss latency, bytes read, read amplification, useful prefetch, memory occupancy and end-to-end throughput where a real model/backend is available.
5. Add unit/property tests and CI coverage for capacity enforcement, eviction correctness, replay determinism and policy comparisons.
6. Keep synthetic results and real hardware/model measurements in separate result categories.

### Gate 5B — Cache, predictor and scheduler ablations

Compare at least: simple LRU or another transparent baseline; frequency/reuse-aware cache; router-aware prefetch; confidence-aware predictive prefetch; no-prefetch and no-prerouter ablations.

Report end-to-end tokens/s and latency, prediction coverage/precision/recall, useful-prefetch ratio, wasted reads, bytes moved/token, page faults, hit/miss latency, cache occupancy, peak memory, CPU utilization and time hidden by prefetch. Hit ratio is diagnostic, not a performance result.

Use a cost-aware scheduler considering confidence, expert size, storage/RAM latency, cache pressure, expected reuse, overlap and miss cost. Compare with simpler policies. Do not assume the prerouter A/B result generalizes.

### Gate 5C — Asynchronous I/O and overlap

Investigate asynchronous I/O, queueing, batching, block alignment and platform mechanisms on every relevant target. Keep a platform-neutral `kiln-io` contract; Linux io_uring/SQPOLL is one implementation to compare with buffered I/O and the target platform's native options. Benchmark syscall/API overhead, throughput, latency tails, CPU use, queue depth, transfer cost, memory contention and overlap. Tier 0 CPU-only measurements remain mandatory, while accelerator-tier tests must include transfer/queue costs and must not assume GPU-serving mechanisms help.

Research block-aligned expert/token sorting: grouping expert requests and aligning block reads may reduce I/O overhead. Treat as a falsifiable hypothesis and measure on the actual model/storage layout.

### Gate 5D — Model correctness and end-to-end acceptance

Validate the exact model/checkpoint, tokenizer and runtime revision. Exact prefetching should not change outputs; approximate expert surrogates can and require quality evaluation. Record model revision, quantization, evaluation data, seeds/settings, output correctness and uncertainty. Measure sustained speed, prompt processing, memory peak, cache behavior, bytes/token, quality and a same-hardware baseline.

**Phase 5 exit gate:** reproducible end-to-end result on the required model/hardware, sustained speed, memory peak, measured data movement, cache/prefetch behavior, quality and fair baseline. A build, microbenchmark or cache-hit result alone does not complete Phase 5.

## 8. KV cache and long-context research

Treat KV-cache placement and compression as a distinct memory-hierarchy problem. Weight streaming does not solve unbounded context growth by itself.

FastStrata/Strata-style hierarchical caching and low-bit KV work may inform placement, transfer granularity and scheduling. A reported 8-bit KV fit for 144K context on 6 GB VRAM is GPU-specific, not a CPU-only KILN capacity guarantee. Separate KV capacity from attention compute, bandwidth and quality.

Research lead: 2026 survey of I/O for LLM inference, Artificial Intelligence Review: https://link.springer.com/article/10.1007/s10462-026-11651-1

Use the survey to structure profiling across weight, KV-cache, activation, metadata and intermediate-buffer traffic. Verify publication details and exact claims before citing publicly.

## 9. Speculation and external MoE-serving research

**SPICE:** “Speculative Prefetching with Low-Rank Expert Surrogates and Heterogeneous Orchestration,” arXiv:2608.21240: https://arxiv.org/abs/2608.21240. Investigate confidence-aware expert prefetch and miss handling. Separate scheduler ideas from approximate surrogate execution. Surrogates can affect quality; test against exact execution and define an error budget.

**llama.cpp speculative decoding:** https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md. The master documentation is mutable. Pin the exact revision and verify flags/semantics before experiments.

### Platform and execution-stack landscape

The sources below are design and experiment leads, not implementation commitments. Hardware support is fragmented: an API can expose CPU/GPU/NPU while requiring different model formats, compiled artifacts, operator subsets or vendor runtimes. Keep the KILN model/API boundary stable and test backend parity, dispatch coverage, artifact lifecycle and CPU fallback for each candidate.

| Project / stack | Technology and framework | What KILN should take or test | Limits / status |
|---|---|---|---|
| llama.cpp / ggml backend registry ([backend docs](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)) | C/C++; runtime backend/device registry, per-backend kernels, dynamic backend loading and device properties across CPU and accelerator backends. | Reuse registry/device discovery and narrow backend integration in `kiln-hal`/the llama shim; retain KILN's placement policy and dispatcher. Verify dynamic loading and multi-backend behavior at the pinned fork revision. | Backend availability does not prove KILN per-layer placement or parity. Upstream docs are mutable; pin revisions. |
| OpenVINO GenAI / llama.cpp OpenVINO ([llama.cpp guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)) | Intel OpenVINO execution for CPU, iGPU, dGPU and selected NPU configurations; GGUF path exists in llama.cpp. | Prototype as an optional Intel backend; record operator/quant coverage, conversion cost, accuracy, device selection and end-to-end decode. | NPU/model coverage and quant support are narrower than the headline device list; treat current support as experimental until verified against pinned versions. |
| AMD Ryzen AI ([Ryzen AI docs](https://ryzenai.docs.amd.com/)) | Vendor stack separates NPU/XDNA flows (commonly ONNX Runtime GenAI/other Ryzen AI software) from llama.cpp's iGPU backend; open IRON explores lower-level NPU development. | Map actual device discovery, supported operators/formats, unified-memory behavior and fallback. Keep AMD NPU and iGPU as distinct backend capabilities. | Do not infer that llama.cpp GGUF support on iGPU implies an NPU GGUF backend. Vendor software/platform support varies. |
| Qualcomm Genie / QAIRT ([Genie](https://github.com/qualcomm/Genie), [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hexagon)) | Genie offers a C-facing on-device stack; llama.cpp has CPU/Adreno/Hexagon paths, while QAIRT workflows can require precompiled target bundles. | Study device enumeration, one API over CPU/GPU/NPU, compiled-artifact caching and fallback behavior. | Model conversion, quantization and supported SoC constraints can bind the NPU path; not a drop-in universal GGUF backend. |
| ONNX Runtime GenAI execution providers ([docs](https://onnxruntime.ai/docs/genai/)) | C++/Python GenAI wrapper over ONNX Runtime execution providers including CPU, CUDA, OpenVINO, QNN and others depending on platform/package. | Evaluate as an optional accelerator adapter or small draft-model path; learn provider selection and graph partition/fallback behavior. | ONNX conversion and provider-specific compiled graph requirements can duplicate artifacts; it does not replace KILN's GGUF MoE streaming core. |
| MLX ([project](https://github.com/ml-explore/mlx)) | Apple-native array/ML framework designed around Apple unified memory and Metal. | Use as an Apple-tier comparison for shared-memory accounting, placement and memory pressure; compare with KILN's Metal/ggml route. | Separate execution engine and model conversion ecosystem; not a reason to fork the user-facing KILN model contract. |
| IREE ([project](https://github.com/iree-org/iree)), MLC-LLM ([project](https://github.com/mlc-ai/mlc-llm)), ExecuTorch ([project](https://github.com/pytorch/executorch)) | Compiler/runtime stacks spanning CPUs, GPUs and embedded/mobile delegates; commonly compile or package target-specific executable/model artifacts. | Borrow kernel compilation, graph partitioning, delegate fallback, memory planning and deployment-matrix ideas; prototype isolated subgraphs or kernels only when profiling identifies a need. | Full-runtime replacement risks duplicating KILN scheduling and model lifecycle. Target artifacts conflict with transparent one-file delivery unless treated as regenerable keyed caches. |
| MDMA ([project](https://github.com/seuros/MDMA)) and Agave ([project](https://github.com/maci0/agave)) | Small Rust/Zig heterogeneous-runtime experiments reporting multiple CPU/GPU backends and, for MDMA, shared-memory admission concepts. | Inspect memory admission that counts unified physical memory once, explicit fallback and execution-vs-memory planes; compare sparse GEMV/SSD ideas with KILN measurements. | Low maturity and limited independent evidence; verify license, correctness, active maintenance and reproducibility before any dependency or code reuse. |

### MoE streaming, prediction and cache projects

| Project / primary source | Technique and evidence | What KILN should take | Limit / experiment |
|---|---|---|---|
| Janas ([README](https://github.com/prabanta-dev/janas), [hardware notes](https://github.com/prabanta-dev/janas/blob/main/docs/hardware.md), [model format](https://github.com/prabanta-dev/janas/blob/main/docs/models.md)) | Dependency-light C runtime; supports CPU, GPU and Intel Core Ultra NPU configurations, streams experts from NVMe, and retains recurring experts in RAM. The reviewed design sizes its RAM cache from memory available when the model opens and benefits from OS page-cache behavior. It profiles hardware configurations and converts supported GGUF inputs to its `.jns` format. | Treat it as the basic streaming/cache control. Reproduce with the same model, cache budget, cold/warm OS cache and workload; compare KILN's byte admission, explicit I/O/prefetch accounting, backend dispatch, artifact lifecycle and fallback. Learn from its startup hardware calibration without turning per-machine results into fixed modes. | The reviewed version does not establish a trained next-layer expert prerouter or FlashMoE-style ML cache replacement. KILN Phase 5.2/5.3 are intended improvements, not Janas features. Its reported metrics and format conversions require independent checks. Check the repository's GPL-3.0-or-later license before any direct reuse. |
| Edge0 ([paper](https://arxiv.org/abs/2609.18063), [repository](https://github.com/Edge0-AI/Edge0)) | Trained prerouter / expert prediction and Recover-LoRA route recovery; published system results include throughput gains with quality costs that vary by model/setting. | Compare trace prediction and layer-ahead exact prefetch against KILN's Phase 5.0/5.2 gates. Keep router replacement and LoRA recovery in a separate approximate mode. | External metrics do not establish KILN gains. KILN's primary path keeps exact original-router execution; measure predictor runtime, wasted reads and quality against pinned baselines. |
| FlashMoE RF cache ([paper](https://arxiv.org/abs/2601.17063)) | ML cache replacement combines recency and frequency signals for SSD-served MoE experts; paper-reported improvements are not KILN results. | Phase 5.3 comparison against LRU/LFU and other policies on the same byte-bounded real traces. | No reusable public implementation was confirmed in this research pass. A different FlashMoE work describes a GPU serving kernel; do not conflate that paper with the RF-cache replacement work. |
| MoE-Infinity and KTransformers ([MoE-Infinity paper/code](https://github.com/TorchMoE/MoE-Infinity), [KTransformers](https://github.com/kvcache-ai/ktransformers)) | MoE-Infinity uses request-level activation traces, group-activation/locality patterns, prefetch and caching for host-to-GPU offload; KTransformers explores heterogeneous CPU/GPU execution and expert offload. | Extract request/sequence-aware trace schemas, overlap and locality ideas; compare against KILN's per-layer dispatcher and measure transfer/queue costs on each relevant tier. | Published MoE-Infinity results are on accelerator/host-memory serving and do not predict SSD/CPU Tier 0 behavior. Use as architecture and trace-policy comparisons, not KILN performance baselines. |
| FlashInfer, LMCache and Mooncake ([FlashInfer](https://github.com/flashinfer-ai/flashinfer), [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake)) | GPU attention kernels or KV-cache/context reuse and transfer systems, rather than cold MoE expert-weight replacement. | Retain for later accelerator and long-context/KV phases; borrow only mechanisms that match a measured KILN bottleneck. | Not Phase 5 expert-cache substitutes. Evaluate under the correct workload and hardware tier; do not expand the current gate based on adjacent terminology. |

### Research leads to retain

| Lead | Technique to investigate | Potential KILN relevance | Caution / next action |
|---|---|---|---|
| MoE-Infinity (2024) | Expert offload, activation-aware caching, CPU/GPU coordination | Expert locality and offload cost | Accelerator-focused; test CPU/SSD path. |
| FloE (ICML 2025) | On-the-fly expert inference, redundancy and prediction | Prediction and transferred-data reduction | GPU results do not transfer directly. https://proceedings.mlr.press/v267/zhou25j.html |
| MoE-APEX | MoE scheduling/prediction | Scheduler/predictor techniques | Read primary source and record exact method. |
| BigMoeOnEdge | Edge MoE inference | Low-resource deployment | Verify implementation and hardware assumptions. |
| Moraine, Colibri, Edge0, PulsarForge | Routing, caching, edge inference and scheduling leads | Candidate hypotheses for measured bottlenecks | Re-read primary papers/repos; do not infer details from names alone. |
| Run:ai Model Streamer | Model-weight streaming | Streaming and I/O patterns | Datacenter-oriented; evaluate laptop complexity. |
| Workload Map, KAI Scheduler, Grove, GPU Fractioning Operator | Workload placement/resource scheduling | Scheduler design and resource-pressure ideas | Extract mechanism, not system wholesale. |
| Osprey, T-SAR/FairyFuse and related serving/compression work | Preserve previously identified technique/source leads | Possible compression, routing, cache or serving hypotheses | Verify exact source, technique and hardware before phase assignment. |

The research log is inclusive. No lead is removed because it targets another tier. Record technique, original hardware, workload, reported metric/baseline, likely KILN bottleneck, cost, risk, phase and reproduction status. “Not currently applicable” is temporary, not a discard decision.

## 10. Quantitative feasibility and reporting discipline

### Dense-versus-MoE bandwidth sanity check (back-of-envelope only)

For scale, a 70B dense model represented at an idealized 2 bits/parameter has approximately **17.5 GB** of raw weight data in decimal units before scales, metadata, alignment and other overhead. If the full weight set had to be fetched from storage for every generated token, 4 tokens/s would imply roughly **70 GB/s** of weight traffic. This is a deliberately simplified upper-bound-style scenario, not a prediction of actual runtime: caching, quantization layout, reuse, compute and the memory hierarchy change the real result.

For a MoE case with roughly 4B active parameters per token at the same idealized 2 bits/parameter, the active-weight equivalent is approximately **1 GB/token**, or **4 GB/s at 4 tokens/s**, before overhead. This does **not** mean the implementation necessarily reads exactly 1 GB from SSD per token: actual bytes depend on expert routing, shared weights, block granularity, cache residency, prefetch, read amplification and overlap.

Use these calculations to frame measurements, not to declare dense inference impossible or MoE feasibility proven. Replace the assumptions with measured effective bytes/token and usable bandwidth on a pinned model/runtime/hardware setup. Record raw model size, effective representation size, storage bytes read per token, cache-served fraction, overlap, peak RAM, sustained tokens/s and output quality separately.


Distinguish total parameters from active parameters; nominal bit width from effective bytes/parameter; model file size from expert bytes fetched/token; advertised storage bandwidth from measured effective bandwidth; cache hit ratio from useful prefetch ratio; prompt-processing speed from generation speed; microbenchmark speed from sustained end-to-end throughput; and model quality from a single narrow metric.

Measure data movement across SSD, page cache, RAM and CPU caches. Track weight, KV-cache, activation, metadata and intermediate-buffer traffic separately where possible. Re-profile after every successful optimization because the bottleneck may migrate.

Feasibility estimates must use effective bytes/token and measured usable bandwidth. If data is reused or prefetched, quantify cache-served fraction and overlap. Do not infer that the 4 tok/s target is achieved or impossible from a back-of-envelope estimate.

## 11. Research log schema

Append every new finding using this template:

- **Title / primary source / version / date**
- **Hardware tier and target workload**
- **Problem addressed**
- **Specific technique (not just system name)**
- **Reported baseline and metric**
- **Expected KILN bottleneck and falsifiable hypothesis**
- **Likely phase**
- **Quality, memory, portability, license and maintenance risks**
- **Smallest experiment that could falsify the hypothesis**
- **Status:** unread / read / hypothesis / implemented / reproduced / not beneficial under tested conditions
- **Evidence:** artifact, benchmark record and commit

A “not beneficial” result is useful and must remain recorded with tested conditions.

## 12. Open work and immediate execution order

1. **Git state repair:** inspect repo, branches, submodule state, uncommitted changes and backups before cleanup. Preserve all work. Previously recorded last known commit was 0d6cacf, working tree intact, with backup at /home/andreipath/Desktop/KILN-backup-20261005-085331/. This is historical context; verify current state before acting.
2. **Licensing review:** separate task covering repository license, llama.cpp/fork obligations, dependencies, models and any future dual-licensing proposal. Do not assume conclusions.
3. **Phase 3.5:** pin/reproduce dense baseline, profile hot path, run isolated prompt-cache/speculation/thread/batch experiments, and measure --n-cpu-moe where applicable.
4. **Phase 5 Gate 5A:** build representative expert-block I/O feasibility benchmark before expanding streaming runtime.
5. **Phase 5 Gates 5B–5D:** run isolated cache/predictor/prefetch and I/O-overlap ablations, then end-to-end model and quality tests.
6. **Representation validation:** verify TQ1.0 effective size and benchmark ternary K/V/RSD separately; evaluate QTEA as a controlled comparison.
7. **Documentation:** update README claims only after verifying the representation or result. Label external paper results clearly.
8. **Training pipeline design:** write docs/training-pipeline-design.md separately. Keep training out of inference critical path unless measurements justify coupling.
9. **Model lifecycle:** harden the initial GGUF discovery/search/pull workflow. Add model-card/license display, gated-repository handling, hardware-aware size estimates, resumable transfers, and an install flow that asks before choosing or downloading a model. Keep model-format/runtime compatibility checks explicit.
10. **Roadmap-wide UI:** follow docs/ui-architecture.md. Build the Simple/Advanced modes over shared runtime/API state; cover model lifecycle, platform-agnostic capability discovery, per-layer planning, memory hierarchy, inference, quality, diagnostics and future workloads through versioned capability contracts. UI controls must reflect actual runtime support and be expanded phase by phase without rewriting the shell.
11. **Productization:** only after runtime, correctness, performance and quality gates are satisfied.

### Not yet proven by this roadmap

- That 70B-class MoE at ≥4 tok/s is achieved.
- That the Qwen3-30B-A3B phase target has been met.
- That predictor hit ratio predicts end-to-end speed.
- That published GPU/TPU results transfer to a CPU-only laptop.
- That nominal 1.6-bit storage equals 1.6 effective bits/parameter.
- That independently measured optimization gains compose.
- That an external system should be vendored wholesale.

### SLM integration decision — complementary local-inference tier (2026-11-06)

**Decision: Adopt small language models (SLMs) as an evaluated, first-class model-size tier, not as a replacement for KILN's 70B-class MoE mission.** Status: **research/benchmark proposal; no new performance or quality result**. Keep Phase 3.5 as the immediate execution priority and Phase 5 CPU-only expert streaming as the central mission; do not insert a blocking new phase.

**Rationale and scope.** Resident 0.6B–4B quantized models may run without per-token model-weight SSD streaming on Tier 0 (Dell Latitude 7490, i7-8650U, 16 GB RAM, no dGPU), enabling practical low-latency local use while the oversize-model architecture matures. An SLM can serve simple classification, bounded structured-output and tool-selection requests, or be tested as a speculative drafter. It cannot substitute for the ≥4 tok/s 70B-class/~4B-active MoE acceptance demonstration or its quality and memory gates. Nominal Q4 parameter-size arithmetic excludes scales, metadata, embeddings, KV cache, runtime and OS memory; measure actual peak RSS.

**Architectural boundary.** The SLM is a model/runtime client, not a new per-layer backend. Keep deterministic authorization, tool schema validation and existing `RuntimeDispatcher` / backend registry authoritative. Any future task-level policy may choose among a resident small model, a mid-sized local model and a streamed MoE, **above** the per-layer dispatcher. Maintain one API contract; fail safe if the selected model cannot meet policy/quality requirements. Do not put an SLM classifier in the inference hot path without demonstrated benefit. AI-generated tool arguments require independent policy validation, not autonomous permission.

**Candidates (verify exact repository revision, GGUF/tokenizer compatibility, license and model card before testing):**

| Candidate | Proposed use | Required comparison |
| --- | --- | --- |
| Qwen3 0.6B Q4 | Simple classification, structured extraction, optional drafter | Rule-based routing and draftless n-gram speculation |
| Qwen3 1.7B Q4 | Resident lightweight assistant and routing | 0.6B and Qwen2.5-1.5B baseline |
| IBM Granite 4.1 3B Q4 | Structured tool-use/agentic evaluation | Strict schema and task-success tests |
| Qwen3 4B Q4 | Mid-size quality/speed comparison | 1.7B and 3B results |
| IBM Granite 4.1 8B (optional) | Quality-oriented comparison only if memory budget permits | Smaller models on same task suite |

Model versions, availability and architectures require fresh verification; inclusion here is **not** a claim of working KILN compatibility or ranking. For agentic work, require benchmarked tool-call correctness, abstention, hallucination rate, argument safety and fallback behavior; do not infer from generic language benchmarks.

**Phase-contained execution and gates:**

1. **Phase 3.5 / baseline first:** pin KILN and kiln-llama commits, model/tokenizer hashes, quantization, compiler, hardware, power and thermal conditions, prompt lengths, sampling settings and baseline model. Reproduce the existing Qwen2.5-1.5B Q4_K throughput before introducing any SLM model changes.
2. **Resident-SLM comparison:** on Tier 0, benchmark cold/warm load, prefill and decode tok/s, time to first token, peak RSS, CPU and SSD bytes read, and sustained thermal behavior for the candidate small models. Test correctness and task quality on fixed prompts, with repetitions and failure cases. Report absolute metrics and a matched baseline.
3. **Speculative decoding A/B:** test no speculation vs supported pinned-revision draftless n-gram vs a compatible small draft model. Record draft overhead, accepted tokens per verification pass, target-verification latency, output distribution/correctness under pinned decoding, total memory, and **net end-to-end tok/s**. Reject the draft model if it slows Tier 0 or compromises resource/quality limits; acceptance rate alone is not a success criterion.
4. **Non-critical-path task routing (post-feasibility research):** compare deterministic routing alone with deterministic routing plus an optional SLM classifier across fixed simple/complex requests. Measure correct routing, harmful under-escalation, escalation cost, model-switch/loading latency, total time to satisfactory result, and privacy/policy safeguards. No automatic routing rollout before quality thresholds and fail-safe fallback are specified and passed.
5. **Phase 5 predictor research (only after Gate 5A):** an SLM used to predict expert demand must outperform a simple predictor in useful prefetch hit/miss cost, bytes/token, peak memory, and sustained model tok/s. Predictor inference must be early enough to overlap expert I/O. Predictor accuracy alone does not justify integration.

**Decision and exit rules.** Keep small-model support when it delivers a repeatable user-visible capability and does not regress existing runtime behavior. Merge speculative drafting or SLM-based routing/prefetch only with measured end-to-end improvements, defined quality/error budgets, CPU fallback, differential tests, and no unacceptable latency/memory/power cost. Preserve negative results. No standalone SLM training, multi-agent framework or general model-routing subsystem before the existing performance and streaming gates warrant it.

**Prior-art and research references (pin versions for implementation):** llama.cpp speculative decoding documentation https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md ; Qwen3 model family https://huggingface.co/Qwen ; IBM Granite family https://huggingface.co/ibm-granite . External model-card claims are not KILN measurements.

## 13. Session log

### 2026-11-06 — Phase 3.5 speculative decoding and Phase 5.0 predictor evaluation batches

- **PR #36 merged** (commit `ac4c0cd9a1e5505186d0f9ab56a438f538d7bae6`, hosted CI passing): draftless n-gram history proposal, bounded at 16 candidates, real target-model batch verification with greedy per-position logits and rejected-tail KV rollback. New opt-in `kiln bench-spec` requires exact generated-token equality against the greedy baseline. A real-GGUF rejection-parity test exists but is ignored in hosted CI. **No Dell acceleration has been measured.**
- **PR #37 merged** (commit `77dc251a76bcaa98b40119366a7aa20a68b425fb`, hosted CI passing): chronological frozen-training expert transition holdout, accuracy@k and coverage with train-only unigram baseline; exposed through `kiln-bench predictor-holdout`. Added `docs/training-pipeline-design.md` as the Phase 5.0 planning prerequisite.
- **PR #39 merged** (commit `4dc01c1f2faaaa8dd1cf623a3aae59221e2c840e`, hosted CI passing): grouped per-token/per-layer active-expert-set predictor evaluation, correct multi-expert accuracy@k, heldout recall and coverage with no future-label training; exposed through `kiln-bench router-holdout`. Conflicting predecessor PR #38 was closed with its branch preserved.
- The two predictor tools are **exploratory software evaluation**, not model-router trace capture or a trained neural prerouter. Real model logs, calibration, same-model baseline, target-hardware predictor cost and Phase 5.0 k=2 ≥80% acceptance remain **open**.
- Phase 3.5 also remains **open** pending actual Dell throughput ≥27 tok/s, verified speculation speed gain and greedy/output quality parity, a pinned Ollama comparison, and thermally stable measurements. Gate 5A and full MoE streaming remain open, with the 70B-class mission unchanged.


### 2026-11-06 — Phase 3.5 native CPU hot-path batch (merged, CI verified)

- **PR #33** merged `0581efe557e1d2728fe8ac6a0f77c5d9ff58f76d`: the pinned llama.cpp implementation distinguishes decode `n_threads` from batch/prefill `n_threads_batch`. KILN's shim now applies the explicitly selected `--threads` CPU count to both. Previously prefill retained an unrelated default, confounding the prescribed Tier 0 2/3/4 thread A/B.
- **PR #34** merged `25fcf16167a8562ca9bbd50d03c2a46235ec510c`: fix native token-to-piece buffers that previously discarded pieces longer than the initial 256-byte scratch space; bounded retry and propagated decoding errors plus required-size tests.
- **PR #35** merged `21fa7fc94b5355549b87bc1772c5d694c4a1994d`: during prompt prefill exceeding the native batch size, request next-token logits only for the final chunk. Earlier chunks retain KV-position updates but no longer request unused outputs. Includes final-chunk boundary tests and a real-model differential acceptance protocol.
- All three PR heads passed GitHub CI before merge. **These are implementation/correctness milestones, not measured performance gains.** Phase 3.5 remains **open** until Dell Latitude 7490 pinned-model tests prove the ≥27 tok/s threshold, model parity, stable memory/thermals and an equivalent Ollama comparison. Run the existing real-GGUF differential gate for long/short prompts and the dense thread/batch/microbatch matrix; do not infer performance from CI.
- No Phase 5 expert-streaming performance or original 70B-class mission acceptance is claimed. Keep the roadmap's existing phase order and storage-feasibility gates.



### 2026-11-06 — Phase 3.5 native greedy decode fast path (merged)

- **PR #32 merged** (commit `8c19f37607eac558cc6d71afde4d323e352a3dcf`) after the native llama.cpp build and all workspace tests passed in hosted CI. It adds an opt-in `ForwardGreedy` operation which selects the first maximum next-token ID inside the pinned llama.cpp shim instead of copying an entire `n_vocab × f32` logits array across the Rust ABI on every decode step.
- Supports `kiln bench --native-greedy` and `kiln run --native-greedy` as **explicit opt-ins**; default sampler and full-logit behavior remain unchanged. Adds token-sequence fingerprint, EOS status, forward/sampling/token-conversion timing breakdowns and CLI argument opt-in tests.
- Real-GGUF native vs Rust greedy parity test is **ignored in hosted CI** and requires the pinned target model on Dell Tier 0. No tok/s gain, parity acceptance or Phase 3.5 completion is claimed before the model-level A/B is executed.
- Unblocked CI native builds by verifying preinstalled Ubuntu CMake/compiler tools rather than performing redundant network-dependent apt installation.
- Next roadmap gate: run same-model, same-prompt, same-thread/batch native vs Rust A/B; retain native mode only if end-to-end throughput improves with identical output and no memory/quality regression. Continue pinned Ollama comparison and speculative-decoding feasibility. Phase 3.5 remains **IN PROGRESS**.



### 2026-11-06 — Continued roadmap implementation: KV correctness and expert-cache memory path

- **Phase 3.5 — PR #29 merged** (commit `12fbcefd35b2866dcc25d521f54b953e421db019`): fix append-only dialogue continuation so a sampled EOS that was not evaluated into native KV state is never injected into the next prompt. Added EOS/context-budget regression tests. Hosted CI passed.
- **Phase 3.5 — PR #30 merged** (commit `a7503c3d2247e92caaeb891d50a2cc4c53e57ed3`): upgrade the already-existing opt-in real-model KV reuse versus full-prefix replay test to five alternating-order trials, exact token-stream parity checks, and median second-turn prefill timing. Hosted CI compiles the ignored real-model test; it does **not** execute on the Dell. No throughput target is claimed.
- **Phase 5 — PR #31 merged** (commit `725a88cf434a987c5ee7f2ab1306d038b4e94b0b` after hosted CI success): expert blocks may be retained as shared `Arc<[u8]>` handles without full-block copy on warm hits, with backward-compatible owned reads and caller-owned buffer copies. Added concurrent-reader, eviction, indexed-length and fail-before-read tests. External shared handles outliving eviction remain **outside cache-membership byte accounting**, so the 15 GB total peak-memory gate is still open, as are llama.cpp expert-streaming integration and end-to-end speed.
- Phase 3.5 and Phase 5 both remain **open**. Do not confuse merged code, CI, or a synthetic I/O test with verified Dell inference speed, SSD bytes/token, expert routing or accuracy.



### 2026-11-06 — Verified Phase 3.5 / Phase 5A implementation status

The following changes are **merged to the default branch**, with CI successful on their corresponding reviewed heads. Their merge status is a code delivery milestone, **not** evidence of model-performance acceptance:

| PR | Phase | Verified implementation |
| --- | --- | --- |
| #10 | 3.5 | Benchmark distribution statistics and environment metadata |
| #14 | 5A prerequisite | Byte-accounted *logical* memory residency and capacity rejection tests; actual RAM allocations/RSS remain distinct |
| #15 | 5A/5B primitives | Bounded indexed expert blob reads, cache trace replay, prefetch simulations and Dell read-only evidence runner |
| #17 | 5A | Conservative I/O probe artifact summary and conditional traffic-budget sanity check |
| #18 | 3.5 | Linux process RSS and lifetime high-water observations around model runs |
| #20 | 3.5 | Dell CPU thread 2/3/4 × batch 256/512/1024/2048 test matrix |
| #16 | 5A | Linux process-attributed physical-storage read counter (page cache and read-ahead limitations apply) |
| #21 | 3.5 | Separates sampled output rate, actual decode-forward rate, and response rate including prefill |
| #22 | 3.5 | Splits prompt prefill at native llama.cpp logical batch capacity with boundary tests |

**Merged implementation:** PR #24 passed CI and merged as `3a83890002de770736c5643bf21f5e30498ce824`. It wires explicit `n_ubatch` through the C shim/runtime/CLI, preserves bounded prompt prefill and decode-rate statistics, and adds an isolated 128/256/512 microbatch sweep plus opt-in pinned-GGUF chunked-vs-larger-batch logits differential test. The latter is **ignored in hosted CI** and needs Dell execution. Older PR #23 was superseded/closed, preserving branch history. **Phase 3.5 remains open.**

**Still required for Phase 3.5:** pinned and repeated Dell model measurements, matching Ollama baseline, profile of remaining speed gap, model-specific prompt-cache correctness, supported speculation A/B and exit-gate evidence. Historical 10.57/26.86 tok/s remain historical, not current results. No ≥27 tok/s claim.

**Still required for Phase 5A:** real Dell SSD block-read experiments using representative expert sizes and model-routing traces, byte-accurate integrated buffers and observed RSS, physical-versus-cached read separation, evidence-based traffic budget, and end-to-end MoE throughput and quality gates. Synthetic cache wins alone do not complete Phase 5.

The roadmap remains the single phase/status authority. Defer unrelated framework/CI convenience improvements and broad model features until the existing acceptance gates are closed. This note is an implementation ledger, not a phase-completion certificate.



### 2026-10-29 — dense/MoE feasibility and synthetic cache experiment

- Recorded the idealized 70B dense / 2-bit estimate (~17.5 GB raw weights; ~70 GB/s at 4 tokens/s if all weights were fetched each token) and the 4B-active MoE / 2-bit estimate (~1 GB active-weight equivalent per token; ~4 GB/s at 4 tokens/s). These are back-of-envelope calculations, not hardware benchmarks or proof of feasibility/impossibility.
- Recorded a synthetic 100,000-access, 128-expert LRU trace: 18.76%, 36.84%, 68.50% and 88.70% hit rate at 4, 8, 16 and 32 expert cache capacities, respectively. Explicitly classified it as simulation only, not a Dell/Latitude or real-model measurement.
- Follow-up still pending: pin trace generator/distribution and seed; implement byte-bounded cache; compare LRU/LFU/adaptive policies; add deterministic trace replay, tests and CI; then evaluate real model traces and end-to-end behavior separately.

### 2026-10-29 — research integration

- Clarified that active parameter count alone cannot establish MoE speed; expert bytes/token, access pattern, cache residency and overlap must be measured.
- Added Phase 3.5 protocol for pinned dense baselines, profiling, speculative decoding, thread and batch tuning.
- Added Phase 5 gates for expert-block I/O, cache/predictor ablations, asynchronous I/O, model correctness and end-to-end acceptance.
- Kept TQ1.0, ternary K/V/RSD, QTEA, KV offload, speculative decoding, MoE streaming and diffusion as distinct tracks.
- Added I/O survey, SPICE, QTEA, FloE and llama.cpp speculative-decoding references, plus MoE-serving and scheduler research leads.
- Clarified that published research is not a KILN benchmark and cache/predictor metrics cannot substitute for end-to-end speed, memory, quality and baseline evidence.
- Immediate execution order remains Git state verification, separate licensing review, Phase 3.5 baseline/profile, then Phase 5 representative expert-block feasibility benchmark.
- This documentation update does not mark an engineering phase complete and does not claim a new performance result.

### 2026-10-29 — runtime safety merge and API honesty fix

- Merged PR #5 after CI succeeded, including the pinned llama.cpp static-library build and workspace tests. Merge commit: 418df7a21901ee5c512a6e09f9e334dd64282c79.
- Audited the API routes and confirmed that `/api/generate` and `/api/chat` returned hard-coded placeholder text as successful responses, while `/api/tags` returned an empty model list.
- Opened PR #6 to return explicit HTTP 501 JSON errors for generation/chat until real runtime-session wiring exists, and added regression tests for that behavior. CI validation is required before merge.
- Updated README to state the current API limitation accurately. This does not mark API inference or any performance phase complete.

### 2026-11-06 — SLM strategy integrated without changing mission

- Added small-language-model support as a complementary local-inference tier, not a replacement for the 70B-class MoE acceptance goal.
- Defined 0.6B–4B resident models and an optional 8B comparison as unverified test candidates, with deterministic policy boundaries.
- Added Phase 3.5 resident-model benchmark matrix, no-speculation/n-gram/draft-model A/B, and optional later task-routing and Phase 5 predictor experiments.
- Required pinned model/build provenance, Tier 0 measurements, end-to-end improvement and quality/authorization gates; no new phase was declared complete and no performance result was claimed.
- Preserved the immediate Phase 3.5 → Phase 5 feasibility order, backend registry hot path and model-level vs per-layer dispatch separation.

### 2026-11-06 — source-level and frontier/theoretical research audit

- At the pre-PR #14 audit snapshot, `InMemoryManager` counted entries rather than bytes. PR #14 later added byte-accounted logical admission and tier moves; it still does not measure physical RSS or page-cache residency. The audit also identified that stock-GGUF compatibility is unproven if expert bytes are relocated.
- Researched MoE-CORE (2026-10), APEX, BITCOS, Q-Strata (EMNLP 2026-10), ZipMoE (ICML 2026), SSD-LLaMA, OLED-MoE and dynamic speculative drafting; retained earlier Edge0/FlashMoE/SPICE leads without claiming them as new.
- Added Phase 5 byte-budget / weighted-paging and SSD traffic lower-bound gates; separate exact routing from approximation, and separate reported external speedups from KILN measurements.
- Prioritized small, falsifiable Tier 0 experiments, licensing checks, measurable precision/quality gates and compatibility proof, not a wholesale runtime replacement or phase reorder.
- No code feature was implemented or validated by this roadmap research update; Phase 3.5 and Phase 5 remain open.

### 2026-11-06 — Phase 3.5 append-only KV path and native context guard

- PR #25 merged after full CI: optional `kiln run --keep-context` appends only new token segments to the live KV context, retains an emitted-but-not-yet-forwarded token, and resets at the configured context boundary. Existing independent-prompt mode remains default; generic `User:/Assistant:` formatting is not model-specific chat-template support. Unit tests and an **ignored, real-GGUF** incremental-vs-full-prefix logit comparison are included. This is an implementation milestone, not a measured Tier 0 speedup.
- PR #27 merged after full CI: the Rust llama.cpp wrapper rejects a batch whose token positions exceed `n_ctx` before any native decode chunk, preventing partial updates when the requested batch is too long. It does not implement context shifting.
- PR #28 proposes a controlled full-reprefill vs cached-second-turn A/B benchmark through the existing runtime dispatcher. As of this note, the PR is **not merged**; it must pass CI and be validated on the Dell before its timing results are accepted.
- Phase 3.5 remains **IN PROGRESS**. The pinned 27 tok/s Dell comparison, model-specific multi-turn correctness, backend hotspot profiling, and supported speculative decoding matrix are still outstanding. Preserve the 70B-class mission and all Phase 5 gates.

## 14. Acceptance criteria for roadmap changes

- Existing rules, phase ordering, completed history and research leads remain unless an explicit evidence-based decision changes them.
- External techniques remain in the research log even when they target another tier.
- Every performance claim has a pinned environment, baseline, method, repetitions and measured result.
- Every approximation has quality tests and an explicit error budget.
- Every phase has an end-to-end exit gate, not only a component metric.
- Completed status is backed by repository, test or benchmark evidence.
- README, design documents and this roadmap agree.
- Session log is updated and changes are pushed successfully.


### 2026-10-29 — API streaming implementation started

- Began a separate feature branch to implement actual token-by-token streaming for `POST /api/generate` and `POST /api/chat` using newline-delimited JSON chunks.
- Added UTF-8 boundary handling so token byte fragments do not split Unicode characters in emitted chunks.
- Added fail-closed regression tests for streaming requests when no model is configured.
- This work is not merged or CI-verified yet. It does not complete the model registry, pull lifecycle, chat-template handling, or full Ollama compatibility, and it does not mark any roadmap phase complete.

## 15. Research and repository audit — 2026-11-06 (new evidence, no phase reordering)

**Scope and status.** Reviewed the complete tracked KILN Git tree (147 entries), major runtime/CLI/API/quantization/I/O/memory files, the canonical roadmap and relevant designs, plus current primary papers and research artifacts. This is a source-level research audit, **not** a successful Tier 0 hardware benchmark, exhaustive manual review of every source line, or a claim that published results reproduce in KILN. The master roadmap remains authoritative. Preserve all existing phases/rules and the historical 2026-10-29 research addendum.

### 15.1 Current implementation and remaining deployment gaps (reconciled 2026-11-06)

1. **Memory admission is byte-accounted logically, but physical residency is not proven.** PR #14 changed InMemoryManager to require registered allocation sizes, enforce logical byte capacities and move reservations atomically. It does not allocate or free the actual expert tensors, measure process RSS, or account automatically for the OS page cache. PR #31 added shared Arc-backed expert blocks; handles that outlive cache membership remain outside the cache-membership budget. Measure the full live allocation/RSS envelope—including caller-held blocks, KV, activations, I/O buffers, metadata and OS reserve—before claiming the <15 GB Tier 0 limit.

2. **The implemented artifact is a sidecar, not the proposed embedded UMF container.** PRs #41–44 add explicit GGUF tensor maps, aligned expert sidecar packing, per-expert CRC32 checks and verified component readers while leaving the source GGUF intact. Mapping requires an explicit model-aware tensor selection; fused expert tensors are not generally decomposed automatically. CRC32 detects accidental corruption, not authenticity. The embedded KUMF extension and automatic kiln convert workflow described in docs/umf-design.md remain proposals. Runtime materialization of sidecar components into llama.cpp inference is not implemented.

3. **Expert I/O/cache tools and async readers exist; real-machine feasibility is still unverified.** PRs #15–17 provide indexed bounded file reads, cache/prefetch replay and Tier 0 evidence/traffic-budget tooling; PR #16 adds Linux /proc/self/io counters with page-cache/read-ahead limitations. PRs #53 and #55 add a Linux io_uring reader and bounded background prefetch with buffered fallback. These testable components do not establish cold physical NVMe bandwidth, SSD bytes per real-model token, total process RSS, or overlap in inference. Run the pinned Dell workload and record those separately.

4. **The streamed-expert runtime seam exists but is not connected to the MoE compute graph.** PRs #47, #49, #52, #53, #55 and #56 implement guarded source-backed expert bytes, prediction rejection, async coordination and a registered MoEStreamBackend. The resident llama.cpp backend deliberately declines MoE byte operations. The actual model router must still supply authoritative expert IDs, and llama.cpp must consume verified tensor components at the router boundary. Do not call cpu_moe_streamed an end-to-end inference backend until that integration passes model correctness and Tier 0 acceptance.

5. **Per-layer heterogeneous dispatch must be demonstrated, not inferred from registered backends.** The standard inference session still runs through LlamaCppBackend; the separate kiln-core operation dispatcher and the streamed-expert byte-operation registry do not yet prove execution placement across CPU, GPU, NPU, APU/iGPU and memory tiers. Require operation-to-backend traces, supported-operation matrices, fallback behavior and differential correctness for every shipped backend.

6. **The current build and the shipping-loader design conflict.** The default linked-llama feature and current build.rs unconditionally compile the C++ shim and link static libraries from the built vendor/llama.cpp submodule, plus Linux-specific stdc++/gomp libraries. loader.rs has path and dynamic-library helpers, but the active FFI/backend calls the linked shim. The dynamic-llama shipping flow in docs/runtime-loading-design.md is not implemented. Resolve this before claiming clean, repeatable cross-platform deployment builds.

7. **The API and model lifecycle are limited.** The API loads one GGUF at startup from KILN_MODEL_PATH; generation is greedy and only a subset of request options is supported. Chat uses generic role-prefixed text rather than a model-specific template. The CLI now has initial local GGUF discovery, `run auto` when exactly one candidate is found, public Hugging Face GGUF search and verified single-file pull; it relies on system curl/SHA utilities, does not handle gated repositories or resumable downloads, does not rank/select among multiple models, and has no curated compatibility-ranked catalog. `plan` remains unimplemented. GGUF parsing is a structural check, not proof that the runtime can load every architecture. Document and test these boundaries before advertising broad model support.

8. **Acceptance evidence and licensing/release qualification remain open.** The 10.57 tok/s dense result and comparison data are historical; Phase 3.5 still needs a pinned current Dell baseline and exit-gate evidence. Phase 5 still needs real-model correctness, sustained throughput, physical memory/I/O and quality evidence. Define the “within 1% of FP16” calculation; complete dependency/submodule/model license review; define the supported platform/backend matrix; and establish clean-machine release, upgrade/rollback and security checks before a public stable release.

### 15.2 New or newly actionable research, ranked by leverage

| Priority | Research (publication) and technique | Relevance and classification | Smallest KILN decision experiment |
| --- | --- | --- | --- |
| **P0** | **MoE-CORE** (October 2026): asymmetric **per-layer** residency allocation, domain-hot initialization, routing-history replacement and cross-layer prefetch; distinct prefill vs decode staging. https://arxiv.org/abs/2610.01950 | New coordinated policy lead; original evaluation targets GPU/NPU host-device transfers, not Tier 0 SSD→CPU. Main execution mode exact; approximate substitution optional and out of exact quality gate. **Research hypothesis**. | Replay real expert traces with equal/unequal byte budgets, LRU/LFU/recency+frequency, domain hints and phase-specific policies; report SSD bytes/token, misses, cache memory, stall time, quality-neutral end-to-end gains. |
| **P0** | **APEX** (August 2026): dynamically size speculative expert prefetch using predicted confidence rather than fixed top-k. https://arxiv.org/abs/2608.11688 | Keep **exact-routing mode** in primary acceptance; paper's high overlap is not KILN's correctness or throughput. Original evaluation uses accelerator/off-chip-memory setting, not Dell SSD. **Research hypothesis**. | Compare no prefetch, fixed k, and confidence-adaptive k+delta against actual expert traces and measured I/O. Require wasted-read budget and miss-cost reduction. |
| **P0** | **BITCOS** (September 2026): zero-density-aware ternary presence bitmap + compact sign vector; theoretical ideal `2-z` bits/weight for zero fraction `z`, excluding format overhead. https://arxiv.org/abs/2609.16338 | Candidate alternative to KILN TQ1_0 five-trit packing. Reported CPU end-to-end gains are **external results**, not Tier 0 results. **Representation/kernels research**. | Measure zero density and distribution on pinned TQ1.0 models; implement format-sidecar prototype and AVX2 decode/GEMV microbench; compare physical bytes including scaling, conversion latency, quality parity and sustained tok/s. Preserve TQ1_0 compatibility. |
| **P1** | **Q-Strata** (EMNLP October 2026): hierarchical, mixed-precision MoE bit allocation optimizing model-level quality instead of independent tensor proxies. https://aclanthology.org/2026.emnlp-main.2414/ | Relevant to accuracy-within-1% gate and heterogeneous expert bit-width; requires offline calibration and conversion workflow, **not** runtime hot-path training. **Research**. | On one supported MoE compare uniform Q4/Q2 and quality-constrained mixed precision at matched effective bytes; measure WikiText-2 plus task quality, model size, hot-expert bandwidth and SSD bytes/token. |
| **P1** | **ZipMoE on-device serving** (ICML 2026): semantically lossless compression with cache-affinity scheduling and theoretical guarantee under its assumptions. https://proceedings.mlr.press/v306/yang26ds.html | Lossless compressed expert cold tier may reduce SSD reads without consuming the accuracy budget, but decode cost and access sizes may offset gain on i7-8650U. **Theory + experiment**. | Implement **offline** sample expert-block compressor/decompressor; measure compressed physical bytes, decompression cycles, loaded-burst throughput and break-even `bytes_saved / measured_bandwidth > decode_time + added_overhead`. Verify exact bytes. Check licenses before reusing code. |
| **P1** | **SSD-LLaMA** (September 2026): per-expert aligned packs, concurrent delivery, tier-coordinated cache and decoupled residency/execution. https://arxiv.org/abs/2609.18110 | Strong architectural analogue but reports 5090/GPU-class systems and 32 GB RAM, not Tier 0 CPU-only. **Transferable technique, not speed claim**. | Prototype aligned expert blobs, bounded coalescing, queue depth and buffered/direct A/B; measure random I/O and end-to-end stall reduction before changing the runtime. |
| **P2** | **OLED-MoE** (September 2026): cache **inter-iteration** expert reuse during block diffusion generation. https://arxiv.org/abs/2609.33385 | Relevant only to future Phase 6 diffusion or model architectures with iterative denoising; not a drop-in autoregressive MoE speedup. **Retain, defer**. | With a real diffusion MoE later, measure expert overlap across denoising iterations and retention versus prefetch. |
| **P2** | **LTD adaptive speculative drafting** (ICLR 2026): optimize actual draft+verify time rather than acceptance length proxy. https://proceedings.iclr.cc/paper_files/paper/2026/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html | Supports a practical adaptation **without** training a new RL scheduler now: vary draft length from measured acceptance and verification cost. **Phase 3.5 research**. | Compare current no-speculation/ngram/small-drafter across dynamic draft lengths with pinned target quality and total latency on Dell. |
| **P1** | **vLLM LES RFC #63666** (open RFC): shared tiered expert pool and lookahead expert scheduling for XPU, with frequency-aware replacement and prefetching layer L+1 while layer L computes. https://github.com/vllm-project/vllm/issues/63666 | Direct comparison for KILN Phase 5.2/5.3: compute uses the actual router while prediction schedules data movement, which fits exact-routing acceptance. **Unmerged proposal; not validated evidence.** | Extract state machine and tier model; replay KILN traces comparing layer-ahead schedule, cache policy and prefetch waste. Do not port XPU-specific assumptions. |
| **P1** | **MoETier** (MIT repository/spec and simulator): data-first tiering model across VRAM/RAM/NVMe; models CPU/GPU/I/O lanes concurrently and next-layer prefetch. https://github.com/sybil-solutions/moetier | Useful trace schema and cost-model ideas for KILN's existing cache/prefetch replay tools and heterogeneous-tier reasoning. **Small, early project; source-level lead only.** | Compare its schema and lane-cost model to current KILN traces; preserve same workload/pinned trace, include shared-memory topology, and check license/provenance before reuse. |

**Previously tracked leads retained:** Edge0's trained prerouter / Recover-LoRA (https://arxiv.org/abs/2609.18063), FlashMoE RF cache (https://arxiv.org/abs/2601.17063), SPICE surrogates (https://arxiv.org/abs/2608.21240), ternary K/V/RSD, and other research in Sections 6–9. Reuse techniques only after exact method and licensing checks; do not rebrand previous research as newly discovered. Edge0's routing replacement and LoRA alter the execution/quality contract; keep **exact routed-expert execution** as KILN's primary acceptance path, with approximation only as a separately evaluated optional mode.

### 15.3 Mathematical/architectural research gates

**SSD time lower bound, before overlap.** For actual SSD bytes fetched per token `B_ssd` and sustained effective bytes/s `BW_eff`, exposed storage time is bounded below by `B_ssd / BW_eff` absent parallel activity. For requested decode rate `r`, the average SSD traffic budget is `BW_eff/r`. At an illustrative measured sustained bandwidth of 2 GB/s and 4 tok/s, the average budget is only **0.5 GB of SSD data per token**; 2 GB/s is *advertised* in the Tier 0 spec, not a guaranteed measured effective random-read rate. If compulsory expert weights exceed this without cache reuse, compression, lower activation, or changed routing, scheduling cannot manufacture bandwidth. Model overlap can hide time but not violate the sustained traffic bound. Track measured `B_ssd`, `BW_eff`, exposed stall, compute time, CPU cycles and peak RAM separately.

**Online weighted paging with predictions.** Expert selection is a variable-byte-size, sequence-dependent cache/admission problem, not simply maximizing raw hit ratio. Compare a byte-bounded policy (LRU, LFU, ARC-like/adaptive, confidence-aware) to a clairvoyant offline trace optimum only as a **diagnostic bound**, not an implementable baseline. Use a combined objective `expected stall + wasted prefetch cost + decompression cost` under hard memory and quality constraints. Reward useful bytes and avoided latency rather than predictor classification alone. Research survey (July 2026): https://link.springer.com/article/10.1007/s10462-026-11651-1 .

**Compression break-even and entropy.** Lossless methods cannot guarantee meaningful compression for every quantized expert; already-quantized weights may be high entropy. Test compression ratio *and* direct SIMD decode overhead on representative real files. For BITCOS, `2-z < 1.6` only if `z>0.4` under the idealized five-trit 1.6-bit packing assumption; real overhead/alignment and per-block probability change the break-even point. Do not treat 1.585 bits as a universal empirical entropy lower bound for non-uniform ternary symbols.

**Accuracy-preserving vs approximate routing.** Separate exact original-router execution plus prefetch/cache (quality-preserving relative to the same quantized model) from route substitution, surrogate expert execution, training-derived prerouters and lossily compressed/quantized alternatives. Any approximate branch must demonstrate the pinned quality gate independently rather than borrow exact-path claims.

**Stop/go after Gate 5A.** The repository now contains Phase 5.2–5.4 cache, guarded-reader, async-I/O and expert-byte dispatch primitives. Do not expand those mechanisms further or wire prediction into the active model path until Tier 0 representative traces and measured physical memory residency show a credible path to the ≥3 tok/s Phase 5 pilot. The current primitives are not end-to-end streamed inference. If the measured path is not credible, preserve the evidence and evaluate any model/tier change through an explicit mission decision.

### 15.4 Recommended immediate order (reconciled to current main; preserves Section 12)

1. Reconcile current-source facts and roadmap statuses; do not treat historical PR/branch notes as open work.
2. Keep Phase 3.5 as the immediate execution priority: reproduce the pinned Dell dense baseline, profile the hot path, and close its measured speed/correctness gate.
3. Complete Gate 5A using the merged expert-read/cache/evidence tools: real Dell reads on representative model expert ranges, physical-versus-cached traffic bounds, actual total RSS including external shared handles, and an evidence-based bytes-per-token budget.
4. Validate the sidecar/tensor map against the pinned target MoE model, including exact component coverage, fused-tensor limitations, byte-for-byte parity and safe failure on malformed artifacts. Keep the original GGUF intact.
5. Only after credible Gate 5A evidence, connect the Phase 5.1–5.4 building blocks to the pinned llama.cpp MoE router and tensor path; run exact-routing, cache/prefetch/I/O ablations and Phase 5D correctness/quality/performance acceptance.
6. Before shipping, resolve the static-link versus dynamic-loader design, define the supported OS/backend matrix, complete licensing review, and add clean-machine packaging, upgrade/rollback and release qualification.

### 2026-11-06 — Phase 5A validated implementation and storage-counter follow-up

- PR #10 (benchmark statistics) and PR #14 (byte-based *logical* memory admission) were merged with CI passing. Their APIs do not prove physical memory RSS or multi-threaded streaming safety.
- PR #15 was merged as `cdc1df90902a1d4965ccddda13b4a237747b143c` after full CI success. It provides the portable bounded expert-file reader, index validation, cache policy replay, simulated prefetch replay and Dell evidence script. The older conflicting PR #12 is superseded and closed, with its branch preserved.
- This follow-up records Linux `/proc/self/io read_bytes` deltas per sequential/random read pattern, with explicit page-cache/read-ahead limitations. These process-attributed storage counters are **not** physical NVMe device telemetry; the tool is not an inference performance test.
- Pending gates: actual Dell results with pinned model/trace; usable page-cache and physical SSD-byte bounds; combined RSS budget including caller-owned expert copies; GGUF/UMF compatibility proof; and real-model sustained tokens/s and quality. No new performance or phase-completion claims.


### 2026-11-06 — Platform-agnostic research integration

- Re-read the authoritative roadmap and incorporated a broad backend/runtime landscape: ggml registry and dynamic backend loading; Intel OpenVINO; AMD Ryzen AI; Qualcomm Genie/QAIRT; ONNX Runtime GenAI; Apple MLX; IREE, MLC-LLM and ExecuTorch; and the MDMA/Agave implementation leads. The roadmap records what to inspect, the stack each uses, and format/artifact, maturity or portability limits.
- Clarified that the Latitude 7490 is KILN's minimum acceptance system. The required CPU-only Phase 5 path is one acceptance slice; the product remains platform-agnostic across CPU, GPU, NPU, APU/iGPU and available memory tiers. Updated the roadmap and README to remove any implication that Phase 5 defines product scope.
- Added Janas as a RAM-cache baseline and recorded that it does not establish the trained multi-layer prerouter or learned cache replacement KILN proposes. Added the vLLM LES RFC and MoETier as unvalidated Phase 5.2/5.3 comparison leads.
- Preserved phase order, all targets and pending gates. These are source-level research integrations only; no code, benchmark result, backend support, or phase completion is claimed.


### 2026-11-06 — Public README refactor

- Replaced the phase/status-led README with a concise public project overview: mission, intended architecture, platform scope, current maturity, links to technical documentation and a development CLI entry point.
- Removed planning-level phase ledgers, benchmark tables, failed-experiment detail, research audits and internal next steps from the front page; the authoritative roadmap remains the home for that material.
- Kept all performance figures clearly described as targets or left in the roadmap's evidence record. No implementation or performance status changed.


### 2026-11-06 — README style revision

- Reworked README after comparing project-native formats used by Janas and llama.cpp: direct project claim, concrete capability list, honest performance table, hardware context, source-build path, and focused documentation links.
- Removed generic product-brief language. Clearly separated implemented repository components from the unfinished MoE streaming and platform-performance goals.
- No phase status or benchmark result changed.


### 2026-11-06 — Current repository and pull-request reconciliation

- Audited all 58 pull requests visible in the repository: **47 merged, 11 closed unmerged, 0 open**.
- Ten closed PRs are explicitly superseded by merged work: #1→#5, #12→#15, #23→#24, #26→#27, #38→#39, #46→#47, #48→#49, #51→#52, #54→#55, and #57 by the later Phase 5.3 async-prefetch implementation. Their branches remain historical; merging them would duplicate or conflict with the landed work.
- PR #19 (fast Python-only CI workflow) is the only closed-unmerged item not marked as superseded. Its branch is technically mergeable, but its closure records the owner’s instruction to defer workflow convenience work until roadmap milestones are complete. Those milestones remain open, so it remains deferred.
- Updated current Phase 5 ledger and engineering-gap sections to distinguish merged sidecar/cache/async/dispatch primitives from the still-missing llama.cpp MoE graph integration and Dell end-to-end evidence.
- Corrected the memory-manager, I/O, dispatch, build-loader and UMF status descriptions against current main. Older dated session notes are retained as historical records.


### 2026-11-06 — Initial local model library workflow

- Replaced the `kiln list` and `kiln pull` stubs with local GGUF discovery and public Hugging Face GGUF search/download commands; added `kiln search` and `kiln run auto` for model discovery and unambiguous local selection.
- Local discovery scans KILN/Hugging Face/Ollama model locations and optional user paths, recognizes GGUF files, and validates them with KILN's GGUF parser. This is structural validation, not a guarantee of runtime architecture support.
- Downloads require an explicit repository and GGUF filename, verify the Hub-provided SHA-256, parse the completed GGUF, and install it only after verification. No automatic model selection or gated-repository auth is introduced.
- The initial transfer uses the system curl and SHA-256 utility; add a native cross-platform transfer layer, resumable downloads, richer license/model-card review and hardware-aware ranking before broad deployment claims.
- Source and documentation changes are on the feature branch; compilation and CI have not been run in this environment.


### 2026-11-06 — Roadmap-wide UI architecture foundation

- Added docs/ui-architecture.md before UI code so future contributors have the product intent, roadmap mapping, information architecture, API boundaries, capability-driven extension model, privacy/security requirements, packaging direction, implementation sequence and acceptance gates.
- Defined the user's two top-level modes—Simple and Advanced—as views over one runtime/session/configuration contract. The UI is scoped across all roadmap phases and hardware tiers, not only today's CPU/CLI/API implementation.
- Recorded the actual current boundary: existing API has streaming chat/generation and one startup-loaded GGUF; CLI model search/pull exists, but model management, dynamic activation, capability/plan/telemetry APIs and browser UI do not yet exist. Phase 5 streaming primitives are not end-to-end MoE inference.
- No UI code or roadmap phase completion is claimed by this documentation step.


### 2026-11-06 — Initial local UI implementation

- Added the bundled Simple/Advanced web shell to the existing API process, with a local streaming chat screen, stop control, per-model browser-local conversation history, active-model/session display and responsive dark theme.
- Added `/api/ui/capabilities` as a versioned snapshot of current API/profile facts. It marks startup-only model loading, missing UI model management, absent accelerator/per-layer controls and future diffusion accurately; no speculative settings are presented as active.
- Updated the UI continuation guide and README with current behavior and outstanding work. Model search/download/activation, settings APIs, runtime telemetry and accessibility/behavior review remain open.
- `node --check crates/kiln-api/static/ui.js` and `git diff --check` pass. Rust compilation and live browser/API behavior have not been verified in this environment; no phase acceptance gate is claimed complete.
- Expanded the Advanced overview to show the API's actual fixed inference defaults (greedy sampling, 128-token chat limit and role-prefixed prompts), and made its cards adapt to available width. Sampling remains reported as limited rather than presented as an editable setting.
- CI exposed a workspace compile error in the just-merged model-library code (`Option::or_else` passed a value instead of a closure). Corrected the fallback to `Option::or`; rerun CI before merge.
