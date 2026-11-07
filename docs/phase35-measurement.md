# Phase 3.5 — dense-path measurement and implementation guide

**Authority:** The [master roadmap](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) owns the phase ledger, target and exit gate. This guide consolidates the individual Phase 3.5 experiment notes; it does not claim the gate is complete.

## Goal and current status

Phase 3.5 is the controlled dense-path baseline and optimization phase. The historical Qwen2.5-1.5B Q4_K measurement and Ollama comparison in the roadmap/README are not current controlled results. The recorded 27 tok/s goal, pinned Dell comparison, repeatability, output parity, thermal stability and profile remain acceptance requirements.

The unified capture workflow is `scripts/run_phase35_suite.sh`; isolated Linux profiling is `scripts/profile_phase35_hotpath.sh`. Check each script's help/current source before running. The suite records model and runtime provenance, dense thread/batch/microbatch settings, Rust/native greedy A/B, bounded n-gram speculative widths, and KV reuse A/B. The Ollama capture helper records a paired comparison, but identical model weights and equivalent conditions still require manual verification.

## Measurement contract

Pin KILN and llama.cpp revisions, build flags/compiler, model and tokenizer hashes, quantization, prompt/context/output length, sampling, thread and batch settings, OS/kernel, power and thermal conditions, cache state, run count and command. Compare paired runs on the same machine and workload. Report median and spread after warm-up and sustained thermal stabilization.

Keep distinct:

- prompt-processing rate, decode-forward rate, sampled output rate, and response rate including prompt work;
- isolated profiling runs and throughput captures;
- measured KILN output and expected/target rates;
- CI smoke-model functionality and Dell target-hardware performance;
- cache hits/predictor acceptance and end-to-end speed.

A capture with missing settings, parity failures, incomplete matrix, unavailable profiling, or unverified model identity is incomplete evidence, not a pass.

## Implemented experiment controls

The suite brings together these opt-in or independently measured paths:

- **Threads and batching:** test the configured thread, `n_batch`, and `n_ubatch` matrix. Decode and batch/prefill thread settings must be interpreted according to the pinned llama.cpp revision.
- **Append-only KV reuse:** compare reuse with full-prefix replay. Token identity and EOS/context-budget handling are correctness gates; prefill savings are not a decode-speed claim.
- **Bounded prefill and final-chunk logits:** chunk prompts within native batch limits; request next-token logits only where needed. These are correctness/resource behaviors, not automatic speed claims.
- **Native greedy path:** opt-in `--native-greedy`; compare exact token sequence with the Rust path before considering speed.
- **Draftless n-gram speculation:** opt-in `--spec-type ngram-mod`; the target model verifies proposals and rejected KV state is rolled back. Require exact output parity and net end-to-end improvement.
- **Memory observations:** process RSS and lifetime high-water measurements are not total system RAM and do not alone satisfy the Phase 5 memory gate.

Run each optimization separately first, then measure combinations after a single change shows a repeatable benefit. Preserve negative results and the exact conditions that produced them. Do not enable experimental paths by default based on a microbenchmark or smoke test.
