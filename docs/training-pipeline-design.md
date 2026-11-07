# Phase 5.0 — expert predictor training and feasibility protocol

## Authority and outcome

The master roadmap owns the phase order and acceptance target. Phase 5.0 asks whether a CPU-feasible expert-demand predictor can achieve at least 80% correct next-expert coverage at k=2 on a pinned MoE workload without future-label leakage. This is a feasibility target, not a Phase 5 completion certificate.

## Trace provenance and event boundaries

Capture real router decisions from the pinned llama.cpp fork and exact model/tokenizer. Record model SHA-256, KILN/fork revisions, quantization, prompts, settings, dataset split, seed, and layer/expert identifiers. Synthetic or self-generated traces do not establish model routing accuracy.

The evaluation must preserve token/layer event boundaries and the full active-expert set for each event. The current grouped router holdout evaluator supports grouped per-token/per-layer sets; the older single-next-expert predictor holdout is a separate exploratory baseline. Do not concatenate simultaneous experts into unrelated sequential labels.

## Leakage-resistant evaluation

Train only on the chronological training segment and evaluate on a later, frozen holdout without updating on holdout labels. Compare with a train-only global-frequency baseline. Report grouped top-k coverage/recall, precision, event coverage, cold start, task mix, sample size, distribution shift, predictor latency/CPU cost, and peak memory. Count missing candidates as misses in overall coverage.

The evaluator's threshold field describes only its input trace and metric. It does not prove an 80% result on the target model, deploy a neural prerouter, or perform physical expert prefetch.

## Training follow-up

If deterministic transition baselines fail or real traces are unavailable, first build an auditable exact-router capture method and verify its labels. Evaluate a shallow CPU-capable predictor with pinned splits and explicit runtime cost. Consider Recover-LoRA or other recovery training only as a separate approximate execution mode with quality budget, privacy/data controls, and measured Tier 0 memory cost. Training must remain outside inference's critical path unless measurements justify coupling.

## Advancement gates

- Complete realistic storage feasibility (Phase 5A) before advancing predictor integration.
- Measure accuracy/coverage, CPU latency, training data size, peak memory, and performance under cold start and distribution shift.
- Compare useful prefetch and end-to-end miss costs, not prediction accuracy alone.
- Preserve exact original-router execution as the primary acceptance path; route substitution or surrogate experts require separate quality acceptance.
- The Dell Latitude 7490 target and all final Phase 5 speed/memory/quality gates are recorded in the master roadmap.
