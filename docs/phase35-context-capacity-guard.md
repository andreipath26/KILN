# Phase 3.5 — native KV context-capacity guard

KILN's llama.cpp shim uses monotonically increasing KV positions and does not currently implement context shifting. Rust previously rejected only i32 position overflow, leaving an ordinary request past `n_ctx` to fail inside native decode (possibly after earlier chunks had been processed).

The Rust wrapper now retains the configured `n_ctx` and checks the **entire incoming token batch** before issuing the first native decode chunk. If the requested end position exceeds the limit, it fails without advancing `n_past` or partially writing any of the new batch.

The guard is conservative and does not imply sliding-window/long-context support. It also does not by itself enforce memory RSS bounds or alter the model's performance.

Unit tests verify one-token overrun and multi-chunk overrun detection without invoking native code; existing integer-overflow and batch-capacity tests remain. Real GGUF prompts at and below the limit still need runtime acceptance testing on Tier 0.
