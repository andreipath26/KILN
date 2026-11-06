# Phase 3.5 — bounded prompt prefill via llama.cpp

## Failure corrected

The runtime previously passed an entire tokenized prompt to `kiln_llama_decode` in a single call. The C shim allocates a reusable `llama_batch` with `n_batch` slots and deliberately rejects `n_tokens > batch_capacity`; legitimate prompts larger than `n_batch` could therefore fail even when they fit within `n_ctx`.

`LlamaContext::decode` now partitions incoming tokens into at most the native batch capacity, feeds each slice to the unchanged C shim, advances `n_past` only after each successful decode, and leaves the final logits from the final chunk available through the existing `forward` interface.

### Acceptance and limitations

- Prompt order and positions are identical to one-token-at-a-time prefill; no prompt reuse or speculative decoding is implied.
- Empty batches, invalid capacities and position overflow fail before touching the native model.
- If a later chunk fails, earlier chunks can already be present in KV state. The caller must `Reset` the session before retrying, rather than assuming the entire prompt was rolled back.
- Unit tests cover the chunk partition/in-order guarantee and zero-capacity validation. **A real GGUF differential test must compare token logits/output with the previous one-chunk behavior for prompts that fit n_batch and validate a longer prompt on Dell.**
- Chunking improves supported prompt lengths and enables more reliable Phase 3.5 test matrices. It is **not** an accepted speedup, prompt-cache retention, or completion of the ≥27 tok/s Tier 0 target.

The next measured Phase 3.5 work remains repeatable Dell CPU thread/batch benchmarking against a pinned Ollama baseline, prompt-cache reuse, and profiling the native matmul/decode hot path.
