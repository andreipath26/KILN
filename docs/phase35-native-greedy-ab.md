# Phase 3.5: native greedy sampling A/B (opt-in)

KILN's baseline greedily samples a Rust-owned `Vec<f32>` after every model forward. Even on CPU-only systems this transfers and allocates `n_vocab × 4` bytes per token across the C ABI and then scans the vocabulary in Rust. The `ForwardGreedy` runtime operation instead selects the same first-max token inside the llama.cpp shim, returning one integer. The existing `Forward` operation and all logits-based sampling modes remain unchanged.

## Controlled experiment

Use the exact same pinned GGUF, commit, prompt, thread/batch settings and thermal state. Both modes run through the same dispatcher and C shim.

```bash
kiln bench /path/to/model.gguf --threads 4 --batch 512 --ubatch 128 \
  --tokens 128 --warmup 1 --runs 5 --json > baseline-greedy.json

kiln bench /path/to/model.gguf --threads 4 --batch 512 --ubatch 128 \
  --tokens 128 --warmup 1 --runs 5 --native-greedy --json > native-greedy.json
```

The report's `sampling_path` identifies which operation was used. Each run also contains `token_fingerprint_fnv1a64` and `eos_encountered` to compare the **exact sampled token sequence** and stopping conditions between modes; a matching digest is a useful regression check, not a mathematical proof of collision-free equality. Compare **total decode-loop time**, **decode-forward rate**, and **end-to-end response throughput** rather than just native sampling overhead. Differences in model quality are not acceptable. The first-max tie behavior matches the existing Rust greedy implementation.

The opt-in differential gate uses an identical pinned model, tokenization, prefill and 32 subsequent decode steps:

```bash
KILN_TEST_MODEL_PATH=/path/to/model.gguf cargo test -p kiln-runtime --lib \
  native_greedy_matches_full_logits_for_real_model -- --ignored --nocapture
```

Hosted CI compiles the test but does not run it because the model is not available. Do not enable native sampling for regular `kiln run` or API inference until target-machine A/B shows measurable end-to-end benefit and the real-model differential gate passes. **No speedup or 27 tok/s gate is claimed by this code change.**
