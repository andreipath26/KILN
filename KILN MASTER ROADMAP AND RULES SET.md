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

## 13. Session log

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


### 2026-10-29 — Phase 5A expert-block I/O implementation

- Merged PR #11 after GitHub Actions succeeded; API generation/chat now support newline-delimited JSON streaming when requested. Full Ollama compatibility remains incomplete.
- Started the Phase 5A feasibility work on a separate branch.
- Added a portable, bounded random-access file reader that reads only requested ranges and rejects out-of-bounds/overflowing ranges before I/O.
- Added expert-index validation for blob-size agreement, sorted layer/expert IDs, non-empty in-bounds ranges, and non-overlapping byte intervals.
- Added a `kiln-bench expert-io` command to compare sequential and deterministic random block reads against a real file, reporting throughput and p50/p95 per-request latency. The OS page cache is explicitly uncontrolled.
- Added unit tests for exact range reads, invalid ranges, empty ranges at EOF, and the I/O probe's two access patterns.
- This is an implementation milestone only. CI must pass before merge; the Dell measurements and the Phase 5A acceptance decision remain pending. No expert-streaming or 70B performance claim is made.


## 15. CPU-only large-MoE execution plan (research-informed, evidence-gated)

### 15.1 Target and non-negotiable measurement rules

Target hardware is the user's Dell Latitude 7490 (Intel Core i7-8650U, 16 GB RAM, no discrete GPU, NVMe nominally around 2 GB/s). These are target specifications, not measured KILN results. Every hardware claim must identify the machine and exact setup; do not substitute hosted CI or another desktop.

Primary stretch target: run a 70B-class MoE with approximately 4B active parameters at >=4 generated tokens/second while staying within the declared RAM budget and keeping WikiText-2 quality within 1% of the pinned FP16 reference under a documented protocol. This remains unproven. Also track the nearer Qwen3-30B-A3B target separately; do not silently substitute one for the other.

Keep these quantities separate in every report:
- total versus active parameters;
- nominal bits versus effective on-disk bytes/parameter;
- full model size versus expert bytes actually fetched per generated token;
- OS page-cache hits versus physical-storage reads;
- advertised versus measured sequential and random-read throughput;
- prefetch requests versus timely, useful prefetches;
- microbenchmark throughput versus sustained end-to-end generation;
- prompt processing versus token generation;
- model quality versus a single narrow proxy metric.

Illustrative lower-bound arithmetic only: 4B active weights at 2 bits/weight are 1.0 GB/token before scales, metadata, shared weights and other overhead. At 4 tokens/s, that is 4 GB/s if all of those bytes must come from storage each token. If half of that traffic is served from RAM, the simplified storage demand falls to 2 GB/s. At 1.58 bits/weight the corresponding idealized values are 0.79 GB/token and 3.16 GB/s at 4 tokens/s before overhead. These are not forecasts; actual expert selection, reuse, cache residency, effective representation size and CPU compute must be measured.

### 15.2 Research leads and how KILN will test them

Research findings are hypotheses to reproduce or falsify, not claims about KILN performance.

- **FlashMoE, arXiv:2601.17063 (Jan 2026), https://arxiv.org/abs/2601.17063.** Reports an SSD-resident MoE expert cache using adaptive recency/frequency and reports higher cache hit rates and speedups on its test platform. KILN hypothesis: recency plus frequency can outperform plain LRU on representative expert traces. Test against LRU and LFU using identical traces, cache budgets, cold/warm conditions and replayed routing sequences; report useful hit rate, bytes read, latency and end-to-end tokens/s. Do not import the paper's speedup claim as a KILN result.
- **MoE-Infinity, arXiv:2401.14361, https://arxiv.org/abs/2401.14361.** Relevant techniques include expert tracing, caching and prefetching/offload. Its GPU-centric setup does not establish CPU-only laptop performance. KILN hypothesis: routing-history-aware prefetch can hide a measurable fraction of expert I/O. Compare no prefetch, simple next-use prediction and history-based prediction; record timely-useful, late, unused and evicted-before-use prefetches, plus I/O/compute overlap and end-to-end latency.
- **Survey of I/O optimization for LLM inference, https://link.springer.com/article/10.1007/s10462-026-11651-1 (published 2026-07-24).** Use as a taxonomy and literature index for weight/KV/activation movement, caching, workload drift and routing-logit prefetch. Any technique selected from the survey must receive a concrete hypothesis, an ablation and a measured exit criterion.
- **BitNet b1.58 2B4T technical report, arXiv:2504.12285, https://arxiv.org/abs/2504.12285; official inference framework: https://github.com/microsoft/BitNet.** This is a distinct native-ternary representation path, not evidence that an existing dense/MoE model can simply be stored at 1.58 bits without conversion or quality cost. Evaluate only with compatible model weights/kernel paths and a quality gate.
- **PowerInfer and SSD/offload research** remain contextual leads only where their GPU assumptions, hardware, license and model format are explicitly checked. No wholesale vendoring based on paper results or headline speedups.

Research log follow-up: add each lead using the Section 11 schema, including exact version/date, tested hardware, metric, risks, falsifiable hypothesis, artifact/commit and status. If full paper or artifact cannot be inspected, mark the lead as preliminary rather than fully reproduced.

### 15.3 Required implementation sequence after Phase 5A

1. **Close the Phase 5A correctness gate.** Review PR #12 for API compatibility, bounds-before-allocation, range-index invariants, tests and documentation. CI must pass. Then measure the command on the actual Dell against a real representative file, recording cold/warm caveats and random/sequential access. Do not call it a storage benchmark if page cache is uncontrolled.
2. **Phase 5B — bounded expert cache.** Implement a cache interface and a deterministic trace-replay harness first. Compare no cache, LRU, LFU and an adaptive recency/frequency policy at identical byte budgets. Enforce a hard memory cap and test eviction, oversized entries, duplicate loads, concurrent access and cache invalidation. Choose a winner only from measured results.
3. **Phase 5C — routing-informed prefetch.** Add an optional predictor interface and trace-driven tests before connecting it to live inference. Compare no prefetch and simple/history-based strategies. Cap in-flight requests and prefetched bytes; cancel or discard stale predictions. Track useful/late/unused prefetch and cache pollution.
4. **Phase 5D — I/O/compute overlap.** Introduce bounded asynchronous reads and reusable buffers only after the synchronous path is correct. Compare synchronous versus overlapped operation on the same traces and workload. Measure queue depth, CPU utilization, bytes read, stalls, p50/p95 latency and sustained generation speed. Do not claim overlap from async APIs alone.
5. **End-to-end model gate.** Integrate the smallest compatible MoE model first, verify tensor/index mapping and deterministic output, then test larger models. Record model hash, tokenizer/template, quantization, runtime commit, OS, CPU/thread settings, RAM, storage, prompt, generation settings, warm-up, repetitions and measurement window.
6. **Quality gate.** Pin the FP16/reference model, dataset version/split, tokenizer, evaluation code, seed and metric. Report the full metric and confidence/variation, not just “within 1%”. Reject performance changes that violate the declared quality or memory limits.
7. **Representation experiments remain separate.** TQ1.0, native ternary weights, ternary K/V/RSD and QTEA each need their own storage accounting, compatible kernels, quality tests and baseline. Do not combine optimizations until each has an isolated ablation; then test interactions.
8. **Only then optimize product claims.** No README or release claim of >=4 tok/s, “no compromise on accuracy”, or specific model-size support without reproducible target-hardware evidence.

### 15.4 Phase 5A implementation notes and known limitations

The current PR #12 adds a portable seek/read range reader, expert-range table validation, an expert-blob reader and `kiln-bench expert-io` sequential/random access patterns. Its benchmark reports latency and throughput and explicitly does not control OS page cache. It is a baseline only: it does not implement direct I/O, io_uring, index-file parsing, cache/prefetch, I/O/compute overlap or end-to-end MoE inference. Dell measurements and formal Phase 5A acceptance remain pending.

Before accepting the implementation, ensure range bounds are checked before allocating the requested buffer. A request larger than the file must fail cheaply instead of attempting an allocation first. Keep public API compatibility in mind when changing `ExpertTable`; search all crate usages before replacing existing shapes.

### 15.5 Status snapshot (2026-10-29)

- PR #11, streamed API generation/chat, was merged on 2026-10-29; GitHub Actions run 40835639935 succeeded. This does not establish full Ollama compatibility or end-to-end model performance.
- PR #12, expert-block I/O feasibility, is open. Latest observed head: `3ab5e17427959faa0df26260ce9093d872d7f042`; CI run 40836786234 succeeded. Earlier intermediate commit 3673eb3 failed CI, then a later head passed; preserve that history and inspect current head before merging.
- Phase 5A is implemented in a review branch, not accepted. The Dell benchmark and review of edge cases remain outstanding.
- Next engineering action: close Phase 5A review findings, then implement Phase 5B cache using trace replay and hard memory bounds. Do not jump directly to a live predictor or claim the target is feasible before these gates.

### 2026-10-29 — research-informed MoE plan and execution update

- Added explicit target-machine context and separated theoretical bandwidth arithmetic from measurements.
- Added FlashMoE, MoE-Infinity, the 2026 I/O survey and BitNet b1.58 as research leads with concrete KILN ablations and portability caveats.
- Ordered implementation as expert I/O correctness/measurement, bounded cache, routing-informed prefetch, I/O/compute overlap, end-to-end model tests, then quality-gated representation experiments.
- Added a prohibition on promoting paper-reported speedups into KILN claims without reproducing them on the target workload.
- No Dell measurement or 70B-class performance result is claimed by this roadmap update.


### 2026-10-29 — first synthetic expert-cache experiment

Added `scripts/synthetic_moe_cache_bench.py`. It deterministically generates 100,000 expert references across 128 experts in two phases, each with a distinct hot set (85% hot-set selection, 15% uniform selection). It compares no cache and LRU/LFU/adaptive recency-frequency policies at capacities of 4, 8, 16 and 32 experts. The default storage model assumes 16 MiB per expert, 100 microseconds fixed latency per miss and 1.5 GiB/s bandwidth. These are tunable assumptions, not measured properties of the Dell or the KILN runtime.

Reproducible command:
```bash
python3 scripts/synthetic_moe_cache_bench.py --requests 100000 --seed 42
```

Observed synthetic results for the fixed seed:
| Capacity | Policy | Hit rate | Modelled storage bytes read | Modelled serialized I/O time |
|---:|---|---:|---:|---:|
| 0 | none | 0.00% | 1,562.50 GiB | 1,051.67 s |
| 4 | LRU | 18.76% | 1,269.33 GiB | 854.34 s |
| 4 | LFU | 12.98% | 1,359.73 GiB | 915.19 s |
| 4 | adaptive | 12.98% | 1,359.73 GiB | 915.19 s |
| 8 | LRU | 36.84% | 986.81 GiB | 664.19 s |
| 8 | LFU | 23.95% | 1,188.22 GiB | 799.75 s |
| 8 | adaptive | 23.95% | 1,188.22 GiB | 799.75 s |
| 16 | LRU | 68.50% | 492.20 GiB | 331.29 s |
| 16 | LFU | 45.12% | 857.44 GiB | 577.11 s |
| 16 | adaptive | 45.12% | 857.44 GiB | 577.11 s |
| 32 | LRU | 88.70% | 176.62 GiB | 118.88 s |
| 32 | LFU | 54.66% | 708.47 GiB | 476.85 s |
| 32 | adaptive | 55.55% | 694.53 GiB | 467.47 s |

Interpretation: this deliberately phase-shifting trace favours recency because the hot set changes halfway through. At 32 cached experts, LRU cuts simulated storage reads by about 88.7% against no cache under this synthetic trace. That is evidence only that the simulator and chosen trace respond as expected—not that real KILN will obtain an 88.7% speedup. The adaptive policy is close to LFU at smaller capacities, so it has not demonstrated an advantage here. Do not select an algorithm from this single trace.

Important limits:
- `modelled_serial_io_seconds` is arithmetic from assumed latency and bandwidth, not elapsed storage time. It excludes CPU compute, real SSD behaviour, kernel/filesystem effects, OS page cache, queueing, concurrency and overlap.
- `simulator_wall_seconds` is only the Python simulation runtime, not model inference time.
- The current trace is synthetic, not collected from a model router. The cache capacity is counted in experts, not enforced by actual byte size.
- Next: add trace fixtures representing different reuse/skew and phase-drift regimes; test the cache policy implementation; then collect routing traces and compare against a real-file I/O measurement. Only then implement the cache in the runtime.
