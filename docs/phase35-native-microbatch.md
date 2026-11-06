# Phase 3.5 — native microbatch (n_ubatch) A/B gate

The roadmap requires controlled comparisons of logical `n_batch` and physical `n_ubatch` on the actual Dell Latitude 7490.

KILN's runtime `RuntimeOp::Load` now carries both parameters through the C shim into llama.cpp context creation. The `kiln bench` CLI accepts `--batch` and `--ubatch`; omitted `--ubatch` uses `min(n_batch,512)`. Values must be strictly positive and `n_ubatch <= n_batch`. The same constraints apply to direct runtime calls. The C shim's reusable token batch is sized by logical `n_batch`, and `LlamaContext` bounds long prompt prefills accordingly.

Example after building KILN with the pinned fork:
```sh
kiln bench /path/to/Qwen2.5-1.5B-Q4_K.gguf --threads 4 --batch 512 --ubatch 128 --tokens 128 --runs 5 --warmup 1 --json
```

**Required experiment:** First pin a stable default against Ollama, then hold model, prompts, threads, logical batch, decode length, warmup and thermal setup fixed while varying microbatch 128/256/512 within the logical batch ceiling. Measure actual tokens/s, prompt latency, peak process RSS, OS page-cache pressure and output parity. Treat increasing microbatch size as an empirical hypothesis, not an improvement. Results from this work cannot be substituted for Dell measurements.

**Limitations:** This adds CPU inference configuration and compatibility; it does not implement n-gram speculation, append-only chat prompt caching, per-layer heterogeneous scheduling, or Phase 5 expert streaming. A successful CI build does not satisfy the ≥27 tok/s Phase 3.5 threshold.

## Real-model batch-boundary differential gate

The Rust test `real_model_chunked_prefill_matches_large_batch_top_token` is **ignored by ordinary CI** because it requires a real GGUF file and target-hardware time. On the Dell, run:

```sh
KILN_TEST_MODEL_PATH=/absolute/path/to/pinned-model.gguf \
  cargo test -p kiln-runtime --lib real_model_chunked_prefill_matches_large_batch_top_token \
  -- --ignored --nocapture
```

The test compares identical tokenized prompts longer than 128 tokens through `n_batch=128` versus `n_batch=512`, both with physical `n_ubatch=128`; checks equal tokenization, same top-scoring next token and bounded maximum absolute logit delta. The numeric limit is a diagnostic gate, not a substitute for the roadmap's model-level quality tests. Preserve the exact model hash, compiled fork revision and test stdout. Do **not** claim model parity from CI's skipped test.

## Fix: selected threads must control prompt prefill too

The pinned llama.cpp revision defines **two different thread pools**: `n_threads` for decode and `n_threads_batch` for multi-token prefill. Previously the shim set only `n_threads`, so the `--threads 2/3/4` Dell matrix could unintentionally run prefill using an unrelated llama.cpp default.

The C shim now explicitly sets both to the same requested thread count. This makes the existing Phase 3.5 thread sweep an interpretable CPU experiment without adding a new public API. Actual latency and performance effects must be measured on Tier 0; this commit is **not** a speedup claim. The new behavior applies to model loads receiving explicit profile parameters; loads without parameters continue using llama.cpp defaults.
