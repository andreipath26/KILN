# Phase 5.0 — Expert predictor training and feasibility protocol

## Authority and outcome

The Phase 5.0 deliverable is not merely a synthetic hit rate: establish whether a CPU-feasible expert-demand predictor can achieve **at least 80% correct next-expert coverage at k=2** for a pinned MoE workload, without reading future router decisions and with a fair baseline. This is a prerequisite for graph prerouting and the Phase 5.1/5.2 streaming design. The original 70B/4B MoE mission and Dell Tier 0 performance gates remain unchanged.

### Input traces and provenance

Obtain a real expert routing trace from the pinned llama.cpp fork with the exact model revision and tokenizer, recording model SHA-256, KILN and fork SHAs, quantization, prompts, generation settings, dataset split, seeds and the relevant target layer/experts. Do **not** treat synthetic or self-generated traces as evidence of model routing accuracy.

A trace is a chronological JSON array of `expert_id` integers (single layer), `[layer, expert_id]` pairs or `{"layer":0,"expert_id":3}` objects. The current evaluator measures **one next-expert event**. Multi-expert-per-token router outputs require grouped event boundaries, explicit top-k precision/recall, and a later evaluator version; concatenating unrelated layer IDs into one stream does not establish k=2 routing precision.

### Leakage-resistant first baseline

```sh
kiln-bench predictor-holdout \
  --trace-file /path/to/real-router-trace.json \
  --train-events 8000 --top-k 2 --json
```

Train only on the first N events. Evaluate the strictly later unseen holdout, **without updating transitions on heldout labels**. Count predictions that did not produce candidates as errors in overall accuracy. Report holdout coverage and conditional precision separately, plus a train-only global-frequency top-k baseline. Data unseen in training is not introduced to the predictor's candidate vocabulary.

The implementation is the existing bounded deterministic transition predictor. It is not a neural prerouter, does not claim to achieve 80% on the actual model, and is not physically prefetching expert bytes. The output's `passed_80_percent_feasibility_threshold` is only an **exploratory metric check for the given trace**, not a Phase 5 completion certificate. Holdout sample size, task mix, cold start, distribution shift and resource pressure must be reported.

### Prerouter training follow-up

If the simple heldout transition model fails or a model-router trace is not available, build a portable training-data capture method around exact router decisions (input hidden states plus target expert labels) and audit routing correctness. Evaluate a shallow CPU-capable predictor using pinned splits and explicit runtime inference costs. Only consider Recover-LoRA distillation if on-device training/data/privacy limits can be satisfied. Never assume recovery training can run within Tier 0 RAM without measuring its memory cost.

### Tier 0 acceptance prerequisites

- Target: Dell Latitude 7490, i7-8650U, 16 GB RAM, no GPU, NVMe device; document power and thermal state.
- Record predictor accuracy@2 and coverage across heldout prompts, CPU inference overhead, training data size, peak memory and inference latency.
- Only advance to Phase 5B/5C after Phase 5A realistic storage feasibility is established.
- End-to-end streamed model must meet Qwen3-30B-A3B ≥3 tok/s, cache hit ≥85%, peak RAM below 15 GB; later original 70B-class mission ≥4 tok/s within 1% WikiText-2 FP16 quality reference.
- On failure, document the Phase 5 feasibility decision and fail-safe resident/dense CPU path, rather than silently weakening the mission target.
