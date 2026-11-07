# Phase 5 — expert streaming implementation and feasibility guide

**Authority:** The [master roadmap](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) owns phase order, research, targets, and exit criteria. This page consolidates Phase 5 implementation notes; it does not assert end-to-end streamed inference.

## Mission and gates

Phase 5 investigates exact MoE expert execution across RAM and storage, with CPU-only Tier 0 as the minimum acceptance slice and other tiers retained in scope. The pilot target and 70B-class mission are distinct; see the roadmap for the exact model, speed, memory, cache, and quality criteria.

Proceed in evidence order: measure representative expert storage/RAM/compute bounds (Gate 5A); compare byte-bounded cache, predictor and prefetch policies on leakage-free traces (5B); evaluate asynchronous I/O and overlap on relevant platforms (5C); then prove correctness, quality, resource use and sustained end-to-end performance in the actual model graph (5D).

For measured effective bandwidth, use effective bytes/token and cold/warm behavior, not advertised sequential SSD bandwidth. Record page-cache effects, random-read latency, bytes fetched, useful prefetch, read amplification, total RSS, thermal state and compute time. A cache-hit or predictor score is diagnostic, not a performance result.

## Implemented components and boundaries

The current main-branch roadmap records these components: validated expert tensor maps and GGUF-preserving sidecars; byte-bounded cache admission and shared byte handles; guarded reads and bounded asynchronous prefetch; Linux io_uring reader with fallback; and `cpu_moe_streamed` dispatch for token-scoped expert-byte operations.

These are distinct implementation milestones. The stock llama.cpp model graph does not yet call the expert operations or materialize sidecar components into inference. Real-model Dell throughput, physical SSD traffic per generated token, total memory under shared handles, model quality and Phase 5 exit criteria remain open. See [UMF and sidecar status](umf-design.md) for artifact details; the [roadmap](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) logs the merged component history.

### Supported-MoE lazy-mmap baseline

The KILN llama.cpp fork's shared loader recognizes the standard routed-expert GGUF tensor types used by supported MoE architectures, including split, merged, gate-up and chunk expert tensors. `kiln run MODEL --lazy-moe-experts` and `kiln chat --model MODEL --lazy-moe-experts` opt into `LLAMA_LAZY_MODE_ON`; `kiln bench MODEL --lazy-moe-experts --json` records the selected mode in its report for paired measurements. The default remains the normal loader mode. Dense models are unaffected. Architecture coverage still follows the pinned llama.cpp fork and its GGUF support; this option does not make unsupported models loadable.

This is a resident-GGUF feasibility baseline. It does not use the KILN sidecar, expert cache, predictor, prefetch queue or `cpu_moe_streamed`, and it does not establish bounded RSS or SSD bytes per token. Compare it against the default loader on the same pinned model and host, recording peak RSS and cold/warm behavior. Treat results as diagnostic evidence, not completion of Gate 5A or Phase 5. The roadmap names Qwen3-30B-A3B as a Tier 0 pilot target; that target does not limit the runtime's architecture scope.

## Predictor evaluation

Evaluate the exact active expert set for each token/layer using real router traces from a pinned model and fork. Keep chronological train/holdout separation; never train on future labels. Group simultaneous experts per token/layer and report coverage, precision/recall, cold-start behavior, distribution shift, predictor CPU cost and memory. The grouped `router-holdout` tool is exploratory evaluation; it is not a trained neural prerouter and does not physically prefetch expert bytes.

The ≥80% at k=2 figure is a feasibility target, not a completion certificate. Predictor work advances only after storage feasibility and must be evaluated against simple baselines and end-to-end miss/wasted-I/O costs. See [training and evaluation protocol](training-pipeline-design.md).

## I/O and cache controls

Keep queues, bytes and speculative reads bounded. A cancellation/epoch guard must prevent stale or uncommitted reads from being used. Compare synchronous buffered reads, portable asynchronous fallback and Linux io_uring under the same pinned traces/model/storage. Measure latency tails, queue depth, useful versus wasted bytes, memory contention and time actually hidden by compute. Linux-specific I/O is one implementation, not the cross-platform contract.

Treat expert caching as a variable-size admission problem. Compare transparent LRU/LFU baselines and confidence/reuse-aware options on the same byte budget and trace. Include caller-owned shared handles in physical RSS even if evicted entries leave the logical cache. Synthetic traces and simulated hit rates must remain separate from real model routing and hardware results.

## Gate 5A indexed expert-read capture

The probe samples real byte ranges from a validated `ExpertTable`; arbitrary fixed offsets are not evidence about expert reads. Build an index for the representative blob/model, then run:

```sh
kiln-bench expert-io --file /path/to/expert-blob.bin \
  --index-file /path/to/experts.index.json --block-bytes 1048576 \
  --requests 256 --runs 5 --seed 42 --json

bash scripts/run_tier0_gate5a.sh /path/to/expert-blob.bin \
  /path/to/experts.index.json [output-directory]
```

The CLI validates the index against the blob, samples sequential and seeded-random offsets within indexed experts, and clips each request to the expert's real byte range. JSON records the input paths/size, indexed range count, sampled `(layer, expert_id)` keys, request-size bounds, requested bytes, latency and throughput. The wrapper records model/blob and index SHA-256 hashes with host/toolchain provenance and runs the block-size sweep. Keep `provenance.txt` beside the JSON reports.

This is a buffered read probe. OS page cache is uncontrolled, and Linux process `read_bytes` includes readahead; neither reported throughput nor that counter is a direct cold-NVMe measurement. It does not measure inference, total RSS, useful overlap, or model quality. Use the capture only as Gate 5A input evidence, and apply the roadmap's storage/RAM/compute gate before expanding runtime integration.

## Stop/go rule

Do not expand or wire these components into the active model path until Gate 5A has credible Tier 0 evidence. Do not declare Phase 5 complete until the exact model graph produces a reproducible end-to-end result with pinned model/runtime, sustained tokens/s, memory peak, storage bytes/token, cache/prefetch behavior, quality, and a fair baseline. If evidence fails, retain it and use the explicit mission decision process in the roadmap.
