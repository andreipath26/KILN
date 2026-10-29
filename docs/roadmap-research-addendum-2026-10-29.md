# KILN Roadmap Research Addendum — 2026-10-29

**Status:** proposed roadmap amendments; research leads, not claims of reproduced KILN results.  
**Applies to:** KILN Master Roadmap and Rules Set v9.0, especially Phases 3.5, 4/4.5, and 5.  
**Governance:** this addendum supplements the master roadmap; it does not replace phase ordering, governance rules, or the inclusive research-log policy. No technique is discarded merely because it targets another hardware tier or comes from an external library. Classify each item by technique, hardware tier, phase, evidence quality, and reproduction status.

> **Repository note:** the master roadmap file referenced by the project is not currently present at the root of this GitHub repository. This file records the new research and proposed roadmap changes without pretending to be a complete replacement for the full master file.

## 1. Decisions and corrections

1. **Make measured data movement the organizing model for Phase 5.** Model weight, KV-cache, activation, metadata, and intermediate-buffer traffic separately across SSD, page cache, RAM, and CPU caches. Re-profile after each successful optimization because the bottleneck can migrate.
2. **Do not infer token speed from active parameter count alone.** MoE active parameters estimate part of compute cost; they do not determine expert fetch volume, random-read latency, cache misses, or synchronization.
3. **Treat the Dell Latitude 7490 target as an acceptance platform, not the product's mission boundary.** Preserve a common runtime contract while permitting hardware-aware policies and kernels.
4. **Keep WikiText-2 within 1% of FP16 as one gate, not a complete quality guarantee.** Specify whether the tolerance is relative or absolute, pin the model/tokenizer/evaluation protocol, and add task-appropriate tests for instruction following, reasoning, coding, multilingual behavior, and long-context retrieval.
5. **A predictor hit ratio is diagnostic, not a performance result.** Report end-to-end tokens/s and latency alongside hit ratio, useful prefetch ratio, wasted reads, bytes moved per token, time hidden by prefetch, cache misses, page faults, CPU utilization, and peak memory.
6. **Treat reported results from external papers as research evidence, not KILN benchmarks.** Keep separate labels for published result, KILN hypothesis, implemented technique, and reproduced KILN result.
7. **Keep quantization, ternary KV, MoE streaming, speculative decoding, and diffusion as distinct research tracks.** They address different bottlenecks and have different quality and implementation risks. Diffusion is not a drop-in acceleration method for autoregressive decoding.
8. **Use a cost-aware dispatcher/scheduler, not only an expert predictor.** Its decisions should consider prediction confidence, expert size, storage/RAM latency, cache pressure, expected reuse, available overlap, and the cost of a miss. Evaluate simpler policies as baselines.
9. **Do not assume optimization gains compose.** Require single-technique ablations before combined experiments and re-profile the resulting system.
10. **Do not claim the 4 tokens/s goal is feasible or infeasible from a simplified bandwidth calculation alone.** Measure real effective bytes/token and bandwidth under the intended access pattern, including the fraction served from cache.

## 2. Phase 3.5 — Dense-path speed run and baseline integrity

### Required measurement protocol

- Pin the KILN commit, llama.cpp/submodule revision, compiler, build flags, model file hash, quantization, tokenizer, prompt, context length, generation length, sampling settings, thread count, batch settings, and power/thermal conditions.
- Compare KILN and a strong pinned baseline on the same hardware and same model/settings. Ollama may be a useful user-facing comparison, but do not treat it as a controlled apples-to-apples baseline unless the backend and settings are established.
- Separate prompt processing (prompt tokens/s) from generation (tokens/s). Include warm-up, repeated runs, median and spread, cold/warm cache conditions, and sustained performance after thermal stabilization.
- Profile time in dispatcher, sampler, decode, dequantization, matrix kernels, synchronization, KV-cache access, and I/O before selecting the next optimization.
- Preserve the reported earlier benchmark as historical until it is reproduced under the protocol; do not silently treat it as a current result.

### Speculative decoding experiment matrix

Compare, one at a time:
1. No speculation.
2. Supported draftless n-gram variants in the pinned llama.cpp revision.
3. A small draft model, if memory and CPU costs permit.
4. Any additional supported speculative method justified by the target model and hardware.

Record accepted draft tokens per verification pass, verification overhead, output equivalence under the specified decoding configuration, peak memory, prompt throughput, generation throughput, and end-to-end latency. A higher acceptance rate alone does not prove a speedup.

### Tuning

- Complete the planned thread-count matrix (including 2/3/4 threads) and batch-size matrix with repeated runs.
- Measure `--n-cpu-moe` systematically where supported and relevant to the pinned backend; document the precise flag semantics for that revision.
- Change one variable per experiment, then test interactions only after individual effects are understood.
- Keep append-only prompt-cache results, n-gram results, and other experiments in separate reproducible records.

**Exit condition:** the dense path has a reproducible baseline, a profile identifying its largest costs, and measured rather than assumed wins for each retained optimization.

## 3. Phase 5 — CPU-only MoE expert streaming

Phase 5 remains the core mission phase. Insert the following feasibility gate before implementing the complete streaming runtime.

### Gate 5A — Measure the storage/RAM/compute ceiling

Create a small benchmark that reads representative expert blocks in the intended on-disk layout and access pattern. Measure:
- sequential and random reads at realistic block sizes;
- cold-cache versus warm-page-cache behavior;
- page faults, read amplification, queue depth, and latency distribution;
- useful versus wasted bytes fetched;
- RAM bandwidth and CPU kernel throughput;
- actual bytes read from storage and RAM per generated token;
- peak resident memory, thermal stability, and synchronization cost.

Estimate the target using measured effective bandwidth and measured bytes/token, not the drive's advertised sequential speed. A simplified bound is:

`storage-limited tokens/s <= usable storage bytes/s / storage bytes read per generated token`

This is only a bound for the storage-served portion. It excludes compute, KV-cache, scheduling, and other overhead. If the measured bound misses the target, determine whether cache residency, improved locality, reduced representation size, a different model, or a revised target is required before expanding implementation scope.

### Gate 5B — Scheduler and expert-cache policies

Compare at least:
- a simple LRU or similarly understandable cache baseline;
- a frequency/reuse-aware policy;
- router-aware prefetching;
- confidence-aware predictive prefetching;
- no-prefetch and no-prerouter ablations.

The scheduler should choose among prefetch, retain, evict, compute, or wait based on expected cost and resource pressure. Track prediction coverage, precision, recall, useful-prefetch ratio, wasted I/O, hit/miss latency, bytes moved/token, cache occupancy, and end-to-end speed. Use the existing recorded prerouter A/B result (20.4 vs 19.9 tok/s) only as a result for that exact experiment; it does not establish a general benefit or scaling behavior.

### Gate 5C — Overlap and I/O plumbing

Investigate asynchronous I/O, queueing, batching, and appropriate platform mechanisms (including io_uring/SQPOLL where supported and beneficial). Benchmark the mechanism on the actual target kernel and storage device. Avoid assuming that an I/O mechanism used in GPU-serving research will improve a CPU-only laptop: isolate syscall overhead, read latency, throughput, CPU consumption, and tail latency.

### Gate 5D — Model and quality validation

- Validate router/expert behavior on the exact MoE model and checkpoint.
- Keep the intended success targets (including the documented Qwen3-30B-A3B target, cache hit target, and memory cap) explicitly labeled as targets until reproduced.
- Run quality tests for any approximation, surrogate, pruning, or compression method. Exact expert prefetch should not change model outputs; approximate expert surrogates can.
- Record model revision, quantization, tokenizer, evaluation data, seeds/settings, and uncertainty for every quality comparison.

**Phase 5 exit condition:** a reproducible end-to-end result with measured sustained generation speed, memory peak, cache behavior, bytes/token, quality, and an apples-to-apples baseline. Do not mark Phase 5 complete based on an isolated cache hit ratio or a microbenchmark alone.

## 4. Quantization and K/V representation research

### TQ1.0 and ternary weights

The project research log states TQ1.0 is already a 1.6-bit representation and no code change is required solely to make that claim. Verify the exact encoding and effective on-disk/on-memory size, and update the README only with a precise statement that distinguishes nominal code width from effective bytes per parameter (scales, metadata, padding, residuals, and buffers).

Evaluate QTEA-style ternary quantization with sparse residual compensation against the existing TQ1.0 path using the same model, tokenizer, hardware, and evaluation suite. A paper's effective bit count or quality/speed result is not transferable without reproduction. Prioritize a target-CPU kernel: storage reduction is not a runtime win if decode, conversion, or lookup overhead dominates.

Research lead:
- QTEA, *Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization*, arXiv:2609.00224: https://arxiv.org/abs/2609.00224

### Ternary K/V and RSD blocks

Keep ternary K/V (including the recorded RSD-block prototype) as a separate Phase 4 research item. Measure effective bytes per token of context, attention latency, perplexity/quality, long-context retrieval, and numerical stability. Do not infer its success from ternary weight results: KV values have different distributions, access patterns, and error propagation.

### KV cache and long context

FastStrata/Strata-style hierarchical caching and low-bit KV research are relevant primarily as data-placement and I/O scheduling ideas. FastStrata's reported 8-bit KV fit for 144K context on 6 GB VRAM is a GPU-specific result, not a CPU-only KILN capacity guarantee. Test on the intended hardware and separate KV memory capacity from attention compute and bandwidth.

Research leads:
- 2026 survey of I/O for LLM inference: https://link.springer.com/article/10.1007/s10462-026-11651-1
- Check the exact paper/version and reported hardware for each Strata/FastStrata result before adding a numerical claim to public KILN documentation.

## 5. External research leads to retain in the research log

These are research leads for reading and targeted experiments, not instructions to import entire systems. Record techniques even when the original target hardware differs.

| Research lead | Technique to investigate | KILN relevance | Limitation / required check |
|---|---|---|---|
| MoE-Infinity (2024) | Expert offloading, activation-aware caching, CPU/GPU coordination | Expert locality, caching and offload cost model | Accelerator-focused; reproduce on CPU/SSD path |
| FloE (ICML 2025) | On-the-fly expert inference, redundancy and prediction | Expert prediction and reducing transferred data | GPU-centric results do not transfer directly |
| SPICE (2026) | Speculative expert prefetch, confidence-aware orchestration, surrogate handling of misses | Cost-aware prefetch and fallback policy | Approximate surrogate can affect quality; evaluate against exact execution |
| QTEA (2026 preprint) | Ternary quantization plus sparse residual correction | Compare with TQ1.0 and CPU kernels | Model-/hardware-specific quality and speed |
| FastStrata / Strata | Hierarchical KV placement and low-bit KV cache | Transfer granularity and memory hierarchy ideas | GPU long-context target; no direct CPU performance inference |
| MoE-APEX, BigMoeOnEdge, Moraine, Colibri, Edge0, PulsarForge | Retain in the inclusive research log with tiers/phases/techniques | Candidate routing, edge inference, scheduling and caching techniques | Re-read primary papers/repos; do not invent or assume implementation details |
| Run:ai Model Streamer, Workload Map, KAI Scheduler, Grove, GPU Fractioning Operator | Streaming, workload placement and resource scheduling patterns | Architecture and scheduling ideas for heterogeneous tiers | Datacenter/GPU systems; extract mechanisms and evaluate cost/complexity |
| Osprey, T-SAR/FairyFuse and related compression/serving work | Preserve the previously recorded candidate techniques and source links | Possible quantization, routing, cache or serving ideas depending on source | Re-verify the primary source and map to a concrete KILN hypothesis |

For every entry, record: primary-source URL, publication/version/date, target hardware, workload/model, claimed metric and baseline, technique, expected KILN bottleneck, implementation cost, quality risk, phase, and status (unread / read / hypothesis / implemented / reproduced / rejected with evidence). “Not currently applicable” is a temporary classification, never a discard decision.

### SPICE research lead

SPICE: *Speculative Prefetching with Low-Rank Expert Surrogates and Heterogeneous Orchestration*, arXiv:2608.21240. It is a lead for studying confidence-aware expert prefetch and miss handling. Evaluate whether its scheduling ideas apply without adopting an approximate surrogate path.

https://arxiv.org/abs/2608.21240

### I/O survey

*I/O for LLM Inference: A Survey of Storage and Memory Bottlenecks*, published in *Artificial Intelligence Review* (2026). Use it to organize profiling across weight, KV-cache, activation, and intermediate traffic. Verify publication metadata and exact statements against the paper before citing them in marketing materials.

https://link.springer.com/article/10.1007/s10462-026-11651-1

### llama.cpp speculative decoding

Use the current documentation as an implementation survey, but pin the revision used for any experiment; the `master` documentation is mutable.

https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

## 6. Research-log entry format

Append each new finding using this schema:

- **Title / primary source / version / date:**
- **Hardware tier and target workload:**
- **Problem addressed:**
- **Technique (specific, not just system name):**
- **Reported baseline and metric:**
- **Expected KILN bottleneck and hypothesis:**
- **Likely phase:**
- **Quality, memory, portability, and maintenance risks:**
- **Smallest falsifiable experiment:**
- **Reproduction status:** not read / read / hypothesis / implemented / reproduced / not beneficial under tested conditions.
- **Evidence and commit/benchmark artifact:**

## 7. Immediate work order

1. Establish the actual Git/worktree/branch state and preserve all existing work before repair.
2. Complete licensing analysis as a separate task; do not mix licensing decisions into performance commits.
3. Establish the dense baseline and profile the hot path (Phase 3.5).
4. Build the representative expert-block I/O feasibility benchmark (Phase 5 Gate 5A) before implementing the complete streaming runtime.
5. Run isolated caching, prefetch, speculation, quantization, and threading experiments with repeatable records.
6. Update public README claims only after verifying representation and benchmark evidence.
7. Write `docs/training-pipeline-design.md` as a separate design deliverable, keeping training out of the inference performance critical path unless measurements justify a coupling.

## 8. Acceptance criteria for roadmap changes

- No technique is deleted solely because it is for another tier or is an external dependency.
- No paper's headline speedup is reported as a KILN result.
- Every performance claim has a pinned environment, baseline, method, repetitions, and measured result.
- Every approximation has a quality test and explicit error budget.
- Every phase has an exit gate tied to an end-to-end result, not only a component metric.
- Phase order and governance rules remain authoritative; this addendum proposes gates and research tasks rather than silently changing completed phase status.
