# KILN MASTER ROADMAP AND RULES SET

**Version:** 9.0 — consolidated research integration  
**Last updated:** 2026-10-29  
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

KILN is a per-layer dispatcher over a unified memory hierarchy. It schedules work across CPU, integrated GPU, discrete GPU, NPU, RAM, NVMe SSD, and optional VRAM as available. Data placement and compute scheduling are coupled but should be profiled separately.

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
| 5 | CPU-only MoE expert streaming | **Core mission phase; full feasibility/end-to-end gates pending** | moe-stream branch of andreipath26/kiln-llama was reported to build on Tier 0; Qwen3-30B-A3B model test waited for data budget. |
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

## 7. Phase 5 — CPU-only MoE expert streaming

Phase 5 aims to stream MoE experts through the memory hierarchy while retaining a useful hot cache, predicting expert demand, and overlapping data movement with computation. Verify implementation and model-specific flags against the pinned andreipath26/kiln-llama revision.

### Existing design targets (not universal guarantees)

Recorded Phase 5 items: UMF container and bit-slicing; multi-layer prerouter plus commit guard; asynchronous I/O investigation including io_uring/SQPOLL where supported; FlashMoE/RF-cache research; registration of cpu_moe_streamed; predictor feasibility ≥80% at k=2; cache hit ≥85%; peak RAM below 15 GB; Qwen3-30B-A3B at ≥3 tok/s. These differ from the top-level 70B-class ≥4 tok/s mission. State exactly which model and target each result addresses.

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

Investigate asynchronous I/O, queueing, batching, block alignment and platform mechanisms such as io_uring/SQPOLL only when supported by the target kernel and storage device. Benchmark syscall overhead, throughput, latency tails, CPU use, queue depth and overlap. GPU-serving mechanisms are not automatically beneficial on a CPU-only laptop.

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
9. **Productization:** only after runtime, correctness, performance and quality gates are satisfied.

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

- Read the tracked repository tree and critical code paths; identified that `InMemoryManager` counts entries rather than actual resident bytes and that UMF stock-GGUF compatibility is unproven if experts are relocated.
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

### 15.1 Engineering gaps discovered in the current repository

1. **Byte-accounted residency is a correctness prerequisite for Phase 5.** `crates/kiln-mem/src/manager.rs` presently defines `InMemoryManager` as an accounting-only prototype; `load` increments `used` by **one per tensor** and `rebalance` is a no-op. It does not enforce actual resident byte budgets, perform SSD reads, or guarantee peak RAM. Gate before streaming integration: record per-expert allocated bytes and alignment; byte-account all resident tensors, IO buffers, page-cache pressure where measurable, KV, activations, metadata and OS reserve; implement bounded admission, eviction/pinning and rollback with tests. Retain the 15 GB Tier 0 overall peak RSS acceptance goal. An expert cache hit count is not a memory-safety guarantee.
2. **Prove format compatibility before UMF implementation.** `docs/umf-design.md` says both that experts are extracted out of the standard GGUF tensor area and that the resulting file loads unmodified in stock llama.cpp. These are not automatically compatible. Build two explicit alternatives and test: (A) complete valid GGUF with separately indexed sidecar expert pack, or (B) a new KILN-only container with documented lack of stock-GGUF compatibility. Do not assert compatibility until standard llama.cpp loads the resulting artifact and tensor offsets/checksums remain valid. Validate crash-safe conversion, partial files, bounds, duplicate/overlap expert ranges and endianness; compare exact tensor values with source.
3. **IO feasibility must come before scheduling complexity.** `crates/kiln-io/src/source.rs` exposes the trait but the main branch does not show a production SSD expert-stream engine; `crates/kiln-models/src/experts.rs` records ranges only. Open PR #12 proposes a bounded expert-read probe, not an end-to-end implementation. Rebase/reconcile against main, pass CI, then run Dell NVMe measurements; distinguish OS-buffered versus direct I/O and warm versus cold page cache, small random reads versus coalesced ranges, P50/P95/P99, queue depth, read amplification and CPU/decompression costs. Default to buffered reads unless the measured alternative wins.
4. **Measure actual per-layer dispatch and fallback rather than infer from type names.** A `RuntimeDispatcher` currently routes session operations through a registered llama.cpp backend, while the separate `kiln-core` per-operation `Dispatcher` is a different abstraction. Require a traced operation→backend map and differential comparison proving when the claimed heterogeneous per-layer dispatch is actually exercised versus delegated to llama.cpp. Do not equate the existence of both dispatchers with fully implemented cross-device selection.
5. **Benchmark and deployment evidence remain distinct.** The `kiln-bench` infrastructure includes synthetic/kernel benchmarks while `kiln bench` provides model-run timings; neither alone closes Phase 3.5 or 5. Open PR #10 improves reporting. Require pinned model hashes, runtime/fork commits, warmup, threads, batches, prompt cache, thermals, median/spread, actual memory/IO and quality. No model-download assumption; respect available Tier 0 storage/data budget.

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

**Previously tracked leads retained:** Edge0's trained prerouter / Recover-LoRA (https://arxiv.org/abs/2609.18063), FlashMoE RF cache (https://arxiv.org/abs/2601.17063), SPICE surrogates (https://arxiv.org/abs/2608.21240), ternary K/V/RSD, and other research in Sections 6–9. Reuse techniques only after exact method and licensing checks; do not rebrand previous research as newly discovered. Edge0's routing replacement and LoRA alter the execution/quality contract; keep **exact routed-expert execution** as KILN's primary acceptance path, with approximation only as a separately evaluated optional mode.

### 15.3 Mathematical/architectural research gates

**SSD time lower bound, before overlap.** For actual SSD bytes fetched per token `B_ssd` and sustained effective bytes/s `BW_eff`, exposed storage time is bounded below by `B_ssd / BW_eff` absent parallel activity. For requested decode rate `r`, the average SSD traffic budget is `BW_eff/r`. At an illustrative measured sustained bandwidth of 2 GB/s and 4 tok/s, the average budget is only **0.5 GB of SSD data per token**; 2 GB/s is *advertised* in the Tier 0 spec, not a guaranteed measured effective random-read rate. If compulsory expert weights exceed this without cache reuse, compression, lower activation, or changed routing, scheduling cannot manufacture bandwidth. Model overlap can hide time but not violate the sustained traffic bound. Track measured `B_ssd`, `BW_eff`, exposed stall, compute time, CPU cycles and peak RAM separately.

**Online weighted paging with predictions.** Expert selection is a variable-byte-size, sequence-dependent cache/admission problem, not simply maximizing raw hit ratio. Compare a byte-bounded policy (LRU, LFU, ARC-like/adaptive, confidence-aware) to a clairvoyant offline trace optimum only as a **diagnostic bound**, not an implementable baseline. Use a combined objective `expected stall + wasted prefetch cost + decompression cost` under hard memory and quality constraints. Reward useful bytes and avoided latency rather than predictor classification alone. Research survey (July 2026): https://link.springer.com/article/10.1007/s10462-026-11651-1 .

**Compression break-even and entropy.** Lossless methods cannot guarantee meaningful compression for every quantized expert; already-quantized weights may be high entropy. Test compression ratio *and* direct SIMD decode overhead on representative real files. For BITCOS, `2-z < 1.6` only if `z>0.4` under the idealized five-trit 1.6-bit packing assumption; real overhead/alignment and per-block probability change the break-even point. Do not treat 1.585 bits as a universal empirical entropy lower bound for non-uniform ternary symbols.

**Accuracy-preserving vs approximate routing.** Separate exact original-router execution plus prefetch/cache (quality-preserving relative to the same quantized model) from route substitution, surrogate expert execution, training-derived prerouters and lossily compressed/quantized alternatives. Any approximate branch must demonstrate the pinned quality gate independently rather than borrow exact-path claims.

**Stop/go after Gate 5A.** Do not expand multi-layer prediction, io_uring/SQPOLL, SLM-based expert prediction or lossless compressor integration until Tier 0 representative storage traces and measured byte-bounded residency show a credible path to the ≥3 tok/s Phase 5 pilot. If not credible, retain the full research record, preserve dense-only Tier 0 usability and evaluate changed model/tier assumptions through an explicit mission decision rather than quietly weakening the target.

### 15.4 Recommended immediate order (does not replace Section 12)

1. Reconcile open benchmarking PR #10 and I/O feasibility PR #12 with current `main`, including their CI and tests.
2. Repair **real byte-accounted** memory management and record a phase-5 peak-memory test before storing large expert blocks.
3. Establish representative SSD-read workload and cache/replacement baselines on Dell; analyze hard traffic budget and compression/overlap break-even.
4. Resolve stock GGUF / UMF compatibility with a round-trip format proof and a selectable sidecar option.
5. Prioritize MoE-CORE variable layer allocation and APEX confidence-adaptive exact prefetch *only if the traces indicate headroom*. Benchmark BITCOS separately in Phase 4.5 and quality-constrained Q-Strata offline.
6. Preserve Phase 3.5 as the immediate inference-performance path and avoid introducing unmeasured multi-model/SLM critical-path behavior.

### 2026-11-06 — Phase 5A validated implementation and storage-counter follow-up

- PR #10 (benchmark statistics) and PR #14 (byte-based *logical* memory admission) were merged with CI passing. Their APIs do not prove physical memory RSS or multi-threaded streaming safety.
- PR #15 was merged as `cdc1df90902a1d4965ccddda13b4a237747b143c` after full CI success. It provides the portable bounded expert-file reader, index validation, cache policy replay, simulated prefetch replay and Dell evidence script. The older conflicting PR #12 is superseded and closed, with its branch preserved.
- This follow-up records Linux `/proc/self/io read_bytes` deltas per sequential/random read pattern, with explicit page-cache/read-ahead limitations. These process-attributed storage counters are **not** physical NVMe device telemetry; the tool is not an inference performance test.
- Pending gates: actual Dell results with pinned model/trace; usable page-cache and physical SSD-byte bounds; combined RSS budget including caller-owned expert copies; GGUF/UMF compatibility proof; and real-model sustained tokens/s and quality. No new performance or phase-completion claims.
