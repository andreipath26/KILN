# Phase 3.5 — final-chunk-only logits during prompt prefill

KILN's Rust llama.cpp bridge splits a long prompt into native `n_batch`-sized decode calls. Previously the C shim requested next-token logits on the last position of **every** batch, including intermediate chunks whose outputs were discarded. That can introduce avoidable logits-output work and memory traffic when measuring prefill of prompts longer than `n_batch`.

The shim's decode entry now accepts an explicit `request_last_logits` flag. KILN sets it to false for intermediate chunks and true only for the final chunk. Native KV positions still advance after each successful batch; the existing greedy/logits methods read the last chunk's outputs only.

## Acceptance

- Hosted unit test checks a 1,025-token input with 256-token batches yields flags `false,false,false,false,true`, and a single-chunk input yields `true`.
- Real-model Dell differential gate: compare next-token argmax, max absolute logit deviation and next tokens on pinned model for prompts both smaller and larger than the logical batch.
- Run long-prompt prefill median time A/B on Dell and measure net speed. **No speed improvement is claimed until hardware validation.**
- The native shim's signature has changed in lockstep with Rust's FFI declaration. This is an internal ABI, not a separately distributed shared library.

This is a Phase 3.5 native prompt-prefill correction, not a new phase or product capability.
