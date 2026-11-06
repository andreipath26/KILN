# Phase 5.4 — streamed-expert dispatcher seam (not end-to-end inference)

The runtime registry now accepts explicit token-scoped MoE expert operations:
- `MoEBeginToken { epoch }`: clear stale speculative blocks and advance the token epoch.
- `MoEPrefetch { epoch, experts }`: submit bounded optional expert predictions to the background worker.
- `MoEPoll`: nonblocking completion drain and staging behind the Commit Guard.
- `MoECommit { epoch, actual }`: return **only exact experts chosen by the authoritative model router** in original router order, with missed predictions loaded from verified expert storage.

`MoEStreamBackend` identifies itself as `cpu_moe_streamed` and can be registered alongside the resident `LlamaCppBackend` in `RuntimeDispatcher`. The native backend explicitly declines MoE-specific operations, so its registered position cannot swallow expert dispatch. A real-file integration test verifies that wrong speculative experts are discarded, misses are fetched from the source, and stale epochs fail closed.

**Critical scope limit:** This is an expert-byte dispatch interface and registry seam. Stock llama.cpp still runs its resident model compute without using these new operations. KILN must wire the fork's actual router/selected expert IDs and tensor materialization into this interface before `cpu_moe_streamed` can be called a production inference backend. The pinned Dell acceptance criteria (Qwen3-30B-A3B ≥3 tok/s, cache ≥85%, RAM <15 GB; original 70B-class mission ≥4 tok/s with accuracy limits) remain open. Registering this adapter is **not** a claim that streamed model inference works yet.

The path is optional and does not alter ordinary dense inference. A missing or invalid streamed expert operation is an explicit runtime error, never silently treated as success by the resident backend.
