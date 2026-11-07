# Phase 5.0 — Expert predictor training and feasibility protocol

## Purpose and execution contract

This document specifies how KILN may train a bounded expert prerouter and an optional recovery adapter. It is a training and evaluation design, not evidence that either component is implemented or useful on Tier 0. Phase 5's primary inference path continues to execute the model's original router and selected experts exactly. Predictions may schedule reads early; they must never select, skip, or replace an expert in the exact path.

Approximate route substitution and Recover-LoRA are opt-in experiments with separate artifacts, commands, reports, and quality gates. They cannot silently become the default when prediction confidence is low or an adapter is absent.

## Inputs and immutable provenance

Every training run has a manifest that pins:

- model repository/revision, tensor and tokenizer hashes, GGUF or source checkpoint hash, quantization, and the matching `ExpertTable`/sidecar hashes;
- KILN and `kiln-llama` commit IDs, compiler/runtime versions, training code version, and feature flags;
- dataset identifiers and license/source records, preprocessing version, prompt/task split, random seeds, and exact train/validation/test example IDs;
- trace schema version, capture command, number of tokens/layers/events, capture host, and whether hidden states are included;
- predictor architecture and weights hash, optimizer and schedule, precision, batch/sequence limits, and peak host/device memory;
- all evaluation thresholds and the script version that calculates them.

Manifests and artifacts are immutable once evaluation begins. A changed model, router, tokenizer, data split, or trace requires a new run identity. Personal or otherwise restricted prompts must not be exported to training storage; use a documented, consented corpus or locally generated prompts and retain only the minimum trace fields needed.

## Router-event capture and data schema

Capture from the pinned target model while the original router and expert dispatch are active. For each token position and routed layer, record the router input hidden state (or a documented fixed-width projection), the original router's selected expert IDs and scores, the layer ID, token position, request/session ID, and model/trace identity. Record grouped top-k expert decisions as one event; do not flatten layer or top-k boundaries into an ambiguous integer stream.

The capture hook is observational: it must not alter router logits, selected IDs, KV state, expert order, or output tokens. Include a parity mode that runs capture on and off with identical deterministic inputs and confirms identical selected IDs and token outputs. If hidden-state storage is too large, store a reproducible projection plus a content-addressed reference to the source run; do not label token-ID-only transitions as hidden-state training data.

Trace partitions are made by source prompt/session and task before training. All events from one session remain in one partition. The intended split is train, validation, and held-out test, with an optional later-time distribution-shift set. Do not randomly split adjacent token events, since that leaks near-duplicate context across partitions. Holdout labels are never used for early stopping, vocabulary construction, calibration, or online updates.

## Predictor training recipe

### Baselines first

For each layer, evaluate the current router's top-1/top-k as the oracle reference, a train-only frequency prior, and a bounded transition predictor. Use the existing deterministic predictor-holdout tooling where its schema fits; treat its trace-event results as a baseline only, not as proof for a neural predictor or live prefetch.

### Portable model candidate

Train a small per-layer or layer-conditioned predictor that consumes only information available before the target layer's expert bytes are needed: a bounded history of preceding router events plus the current layer's available hidden-state projection. Predict a ranked set of at most `k` expert IDs for the next relevant layer/event. The implementation must specify whether it predicts the next layer, a fixed number of layers ahead, or the current layer before the original router decision. Never use the target event's router logits/labels as an input feature.

Use a reproducible supervised objective over grouped expert labels. For top-k routing, use a multi-label loss or a listwise objective that matches the desired candidate ranking. Train with a fixed, versioned optimizer/schedule and a small architecture search bounded by the manifest. Select one checkpoint using validation data only. Calibrate confidence on validation data only and expose a conservative abstain threshold; an abstention means “schedule nothing” and leaves exact demand loading unchanged.

For every candidate, measure held-out recall/coverage@k, precision@k, per-layer performance, calibration/error by confidence bucket, cold-start behavior, and distribution-shift performance. Also report predictor execution latency and CPU time, feature-transfer/copy cost, model bytes, and peak resident memory. Predictor utility is decided by useful completed prefetches and net end-to-end impact after wasted bytes and contention, not accuracy alone.

## Recover-LoRA experiment

Recover-LoRA is a separate recovery mechanism for an explicitly approximate routing experiment. Its training target is the original model's output under the original router on the same token context. The recipe must pin the adapter insertion points, rank, target modules, base quantization, teacher/student precision, optimizer, training tokens, and adapter bytes in the manifest. Train only against the training partition; use validation for early stopping and adapter selection; reserve held-out prompts and a shift set for final quality evaluation.

Compare the adapted approximate route with both (a) the exact original-router model and (b) the same approximate route without the adapter. Report task/language slices, perplexity where applicable, output agreement, failure cases, adapter memory/load cost, and any throughput impact. Recover-LoRA cannot be claimed to restore exactness from a small aggregate score. It must pass a separately approved quality threshold for every supported model/quantization combination. On adapter load, compatibility, numerical, or confidence failure, disable the approximate mode and return to exact routing; if exact routing is unavailable, stop with a clear error.

No adapter training or gradient computation is part of inference. Training artifacts are optional and independently versioned. Shipping an adapter requires documenting its license, base-model compatibility, provenance, reproducibility command, and removal/disable procedure.

## Tier 0 compute and resource limits

The acceptance host for Tier 0 is the Dell Latitude 7490 configuration recorded in the roadmap. Training must not be assumed feasible on that host. Before proposing on-device training, measure available RAM under the OS, swap policy, CPU frequency/thermal limits, training examples/second, total wall time, peak RSS, disk use, and whether the host remains usable. Never use the inference acceptance memory budget as though it were all available to training.

The default plan is to train and select artifacts on a separately provisioned development host, then measure inference-time predictor/adapter costs on Tier 0. If a lightweight fine-tuning run is proposed on Tier 0, require a dry-run memory estimate, a bounded maximum step/token count, checkpoint/resume support, free-space check, and a hard abort on memory pressure. No training step may modify the pinned base model or user model files in place.

At inference, account separately for predictor weights, hidden-state features, temporary activations, adapter weights, adapter activations, cache-resident experts, returned expert buffers, KV cache, and runtime overhead. Report total process RSS and system available memory; a cache budget by itself is not a peak-memory measurement.

## Evaluation and decision gates

1. **Capture validity:** capture-on/off parity passes; trace schema validates; hashes and session-level splits are complete; no split leakage is detected.
2. **Predictor feasibility:** the held-out target workload reaches the roadmap's predictor threshold of at least 80% next-expert coverage at `k=2`, while reporting precision, confidence, layer slices, shift results, runtime cost, and memory. This gate is necessary for considering prerouting, not an acceptance result for streamed inference.
3. **Prefetch value:** on the same pinned trace and byte budget, compare no-predictor/no-prefetch, simple transition, and trained predictor conditions. Count timely useful reads, late reads, unused bytes, bytes/token, cache displacement, CPU/I/O contention, total RSS, and end-to-end tokens/s. Promote only if the net result improves without changing exact outputs.
4. **Recovery quality:** evaluate Recover-LoRA only for the separately named approximate mode. Compare exact, approximate, and approximate-plus-adapter outputs on held-out tasks and shift data. Gate each model/quantization separately; do not pool results to hide regressions.
5. **Tier 0 readiness:** load and run the selected inference artifact on the Dell, including cold start and multi-turn requests. Confirm predictor/adapter failures fall back or stop as specified and do not corrupt exact routing.

Every result is a machine-readable report linked to its manifest. Failed, incomplete, or underpowered runs remain visible and cannot be summarized as a pass.

## Definition of done

The Phase 5.0 predictor-pipeline prerequisite is ready when all of the following are true:

- the event schema and capture hook preserve the original router's exact decisions, with capture-on/off parity evidence;
- manifest validation, session-level splitting, leakage checks, and deterministic training/evaluation commands exist;
- train-only frequency and transition baselines plus a bounded predictor candidate can be compared on held-out grouped top-k events;
- reports include coverage/precision, per-layer and shift behavior, calibration, CPU latency, bytes, RSS, and provenance;
- Recover-LoRA has a reproducible, isolated training/evaluation path and cannot be enabled implicitly;
- inference-side abstention, exact-routing behavior, error handling, and fallback are specified and covered by tests;
- the runbook distinguishes readiness of the pipeline from passing the predictor threshold and from completing Gate 5A/5D or the overall Phase 5 mission.

Passing this document's pipeline gate authorizes controlled predictor experiments only. It does not authorize replacing the original router, declare the Dell capable of training, or establish Phase 5 completion.
