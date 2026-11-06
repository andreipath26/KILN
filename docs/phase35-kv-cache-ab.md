# Phase 3.5 — second-turn prefill A/B test

The master roadmap requires a controlled comparison of **re-prefilling the complete prompt prefix every turn** against **appending only new tokens to a preserved KV cache**. The optional `kiln run --keep-context` mode implements the latter; this `bench-kv` command measures one specific benefit under a reproducible token-identical setup.

```sh
kiln bench-kv /absolute/path/to/pinned-model.gguf \
  --first-prompt "Name two planets." \
  --assistant-fragment " Mars and Venus." \
  --second-prompt "Name one more." \
  --runs 5 --threads 4 --batch 512 --ubatch 128 --json
```

The benchmark tokenizes three segments once, and evaluates the **same exact token IDs** in each mode:

- **Full prefix:** reset the native context and evaluate all three segments in one complete prefill.
- **Append-only:** reset; evaluate the first turn and the fixed assistant continuation; time only the new second-turn segment.

Both modes warm up before recording results. Measured run order alternates to reduce cache/thermal ordering bias. The JSON output contains per-run milliseconds, median/min/max for each mode, reused/new token counts, maximum absolute logit differences and top-token agreement. This makes it possible to identify whether actual prefix caching reduces second-turn prefill latency **without assuming** model quality or tokenization invariance across arbitrary text boundaries.

**Comparability:** The cached timing deliberately excludes the *already completed* first-turn work; it measures latency experienced after the previous turn, not a full-conversation total. This is **not** a decode-throughput test and cannot satisfy the ≥27 tok/s dense gate. The generic role delimiters are not model-specific ChatML. Preserve model SHA-256, fork/runtime commits, compiler settings, temperature/sampling, OS, thermal and page-cache conditions; run the separate decode benchmark and Ollama comparison to close the exit gate. CI can only compile/test without the real model, so the Dell result remains pending.
