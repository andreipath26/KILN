# Phase 3.5 — draftless n-gram speculative decoding, pinned fork

This implementation uses the pinned `andreipath26/kiln-llama` revision's public llama.cpp API, with a narrow C shim rather than vendoring its full server or common code. The fork contains `common/ngram-mod.cpp` and `docs/speculative.md`, but those modules are **not linked** into KILN's static `libllama.a` seam. KILN implements a bounded exact-history matching proposal generator and performs **real multi-position greedy verification in the target model**, with rejected KV positions removed through `llama_memory_seq_rm`.

## Protocol and invariant

1. Target prefill produces the first greedy next-token prediction.
2. The drafter examines a bounded token history (up to 8,192 recent tokens), finds the most recent matching trailing `n`-gram, and proposes at most 16 previously observed continuation tokens, excluding EOS. No standalone draft model or network call is used.
3. The native verifier checks the first proposal against the existing target logits **before evaluating any draft**. A mismatch makes no KV changes.
4. When the first proposal matches, one `llama_decode` verifies the complete proposal with per-position logits. On a later mismatch, the native KV-memory API removes every rejected/unverified position. The accepted prefix and corrected next target token are returned.
5. Only accepted draft tokens and the target greedy correction/bonus token may be emitted. A correction/bonus token is unevaluated until the next forward pass, so KV state and token sequence remain aligned.

Verification errors fail closed and attempt a session reset. The ordinary `kiln bench` path remains unchanged; speculative mode is opt-in in a separate benchmark.

## Exact A/B

```sh
kiln bench-spec /path/to/Qwen2.5-1.5B-Q4_K.gguf \
  --prompt "red blue red blue red blue red blue" \
  --tokens 128 --runs 5 --warmup 1 \
  --threads 4 --batch 512 --ubatch 128 --ngram 3 --draft-max 8 --json
```

This command alternates baseline-greedy and speculative runs on **the same loaded model** and demands identical token IDs/EOS conditions in each trial. It reports generated tokens, proposed/accepted drafts, batch verification calls, forward calls, median generation throughput and relative rate. The timing excludes model load and prompt prefill. A low-repetition prompt may produce no draft proposals and no speed benefit.

### Interactive CLI opt-in

The verified n-gram path is also available during interactive CLI inference:

```sh
cargo run -p kiln-cli -- run /path/to/model.gguf --spec-type ngram-mod
```

The mode is opt-in, uses the same bounded history drafter and target-model verifier, and works with `--keep-context`. It fails closed on invalid verifier metadata. Do not combine it with `--native-greedy`; speculative verification already uses the native greedy target path. The normal `run` path remains unchanged by default. This is a functional integration, not evidence of a speedup; retain it only if the pinned Dell A/B shows a repeatable end-to-end gain with exact greedy token parity.

The ignored real-model CLI parity test exercises two append-only conversation turns through the ordinary and speculative interactive paths:

```sh
KILN_TEST_MODEL_PATH=/path/to/pinned-model.gguf \
  cargo test -p kiln-cli --test interactive_speculative_parity -- --ignored --nocapture
```

CI runs this test against its verified SmolLM2 smoke GGUF. That verifies live CLI integration on a small supported model; it does not substitute for the pinned Tier 0 performance and model-parity gates.

### Real-model rollback parity test

```sh
KILN_TEST_MODEL_PATH=/path/to/pinned-model.gguf \
  cargo test -p kiln-runtime --lib real_model_draft_rejection_restores_identical_kv_continuation \
  -- --ignored --nocapture
```

This checks a forced wrong second proposal, accepted-prefix count, corrected token, native KV position and the following target next-token equality. **Hosted CI compiles but does not execute this model-dependent test.**

No native speedup, stable memory, quality parity beyond deterministic greedy, 27 tok/s target, or Dell acceptance is claimed until the pinned experiment has been executed on the Dell Latitude 7490. If batch verification regresses target throughput, disable it and retain the ordinary greedy runtime path; Phase 3.5's measured-gain rule still applies.
