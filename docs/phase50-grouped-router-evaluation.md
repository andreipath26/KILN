# Phase 5.0 — Grouped MoE router prediction feasibility

An autoregressive MoE router can activate **multiple experts per layer per generated token**. A flat list of individual expert accesses is insufficient to evaluate its top-k active-expert set correctly. This module and CLI evaluate the complete expert set associated with each `(token_position, layer)` event.

## Pinned trace schema

Chronological JSON, sorted by token then layer, with no repeated event or repeated expert within an event:

```json
[
  {"token_position":0,"layer":0,"active_experts":[2,5]},
  {"token_position":0,"layer":1,"active_experts":[4,6]},
  {"token_position":1,"layer":0,"active_experts":[1,5]},
  {"token_position":1,"layer":1,"active_experts":[3,7]}
]
```

Only feed observations from exact model router decisions at inference time. A trace lacking model/tokenizer hashes, real prompts, session boundaries, router layer interpretation, checkpoint and backend provenance does not constitute acceptance evidence. The top-k parameter must be at least as large as the number of experts selected by each event; the evaluator fails rather than reporting impossible k=2 accuracy for a k=6 router.

## Train and holdout

```sh
kiln-bench router-holdout --trace-file router.json \
  --train-until-token 8000 --top-k 2 --json
```

The training split is **by token position**, never by individual event, avoiding same-token leakage across layers. It trains a simple per-layer previous-active-expert to next-active-expert transition baseline on the training prefix. At holdout time, earlier observed decisions can inform later causal predictions, but no heldout target is added to the frozen transition statistics. It reports:

- **Accuracy@k:** fraction of heldout router events where *every active expert* is covered by the proposed top-k set (unpredicted events count as misses).
- **Coverage:** fraction of events with nonempty candidates.
- **Active-expert recall:** fraction of individual active experts included in proposed sets.
- A training-only layer-specific expert-frequency top-k baseline.

This does **not** implement a hidden-state neural prerouter and does not prove the roadmap's 80% k=2 gate on the pinned MoE checkpoint. Correctness tests use synthetic grouped events and verify layer separation, future-label isolation and malformed-input rejection. The actual router trace, hardware costs, training-data budget, inference overhead and real-model acceptance remain open.
