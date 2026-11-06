# Phase 3.5 — Append-only interactive KV reuse

The roadmap identifies repeated prompt-cache resets and re-prefill as a dense-path bottleneck. The `kiln run` CLI previously executed `RuntimeOp::Reset` before **every** prompt and discarded all session context. This implementation introduces an **opt-in**, bounded append-only dialogue mode:

```sh
kiln run /absolute/path/to/model.gguf --keep-context --max-tokens 128
```

The existing `kiln run` behavior remains unchanged without `--keep-context`: each input is an independent raw prompt, and the native KV cache resets every turn.

With the option enabled, KILN creates a token stream using a simple `User: ...\nAssistant:` delimiter (not a model-specific chat template), persists the actual token count forwarded to llama.cpp, and keeps at most one emitted-but-not-yet-forwarded token. Subsequent turns submit only that pending token and the newly tokenized user segment. **Previously executed tokens are not retokenized or re-prefilled.** This avoids claiming a text-based prefix match that cannot always be reproduced by BPE tokenization at segment boundaries.

When the accumulated context plus the conservative maximum next response would exceed the configured `n_ctx`, the code resets the native session and discards previous KV context before processing the new turn. A single turn exceeding the context limit fails safely with a useful error.

## Acceptance requirements

- Unit tests cover exact append behavior, a pending final token, safe reset at context limit, rejection of oversize prompts, and exact-fit boundaries.
- Verify multi-turn correctness and native KV parity using the pinned GGUF on the Dell; CI unit tests do not establish this.
- Compare single-turn baseline and `--keep-context` across two or more turns with exactly controlled prompt/task conditions. Measure *total tokenized/prefilled tokens*, latency to first output, sustained decode rate, and peak RSS. Do not infer a speedup just from avoiding `Reset`.
- Do not equate generic role delimiters with Qwen/ChatML-specific templates or claim that `--keep-context` creates a complete chat product. Structured templates, EOS handling, context truncation and multi-turn quality require model-specific follow-up validation.

This is an opt-in Phase 3.5 runtime change; it neither modifies phase 5 streaming nor changes the original Tier 0/70B mission acceptance targets.

## Observe the prefill savings

Set `RUST_LOG=debug` when running `kiln run --keep-context` to record the new prompt's actual forwarded token count, previously cached native tokens and prefill duration. These are diagnostic counters, not a proof of tokens/s speedup or model-template correctness. The context-window reset always restarts with a fresh first-turn delimiter.
