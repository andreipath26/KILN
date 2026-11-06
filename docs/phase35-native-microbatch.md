# Phase 3.5 — native microbatch (n_ubatch) A/B gate

The roadmap requires controlled comparisons of logical `n_batch` and physical `n_ubatch` on the actual Dell Latitude 7490.

KILN's runtime `RuntimeOp::Load` now carries both parameters through the C shim into llama.cpp context creation. The `kiln bench` CLI accepts `--batch` and `--ubatch`; omitted `--ubatch` uses `min(n_batch,512)`. Values must be strictly positive and `n_ubatch <= n_batch`. The same constraints apply to direct runtime calls. The C shim's reusable token batch is sized by logical `n_batch`, and `LlamaContext` bounds long prompt prefills accordingly.

Example after building KILN with the pinned fork:
```sh
kiln bench /path/to/Qwen2.5-1.5B-Q4_K.gguf --threads 4 --batch 512 --ubatch 128 --tokens 128 --runs 5 --warmup 1 --json
```

**Required experiment:** First pin a stable default against Ollama, then hold model, prompts, threads, logical batch, decode length, warmup and thermal setup fixed while varying microbatch 128/256/512 within the logical batch ceiling. Measure actual tokens/s, prompt latency, peak process RSS, OS page-cache pressure and output parity. Treat increasing microbatch size as an empirical hypothesis, not an improvement. Results from this work cannot be substituted for Dell measurements.

**Limitations:** This adds CPU inference configuration and compatibility; it does not implement n-gram speculation, append-only chat prompt caching, per-layer heterogeneous scheduling, or Phase 5 expert streaming. A successful CI build does not satisfy the ≥27 tok/s Phase 3.5 threshold.
