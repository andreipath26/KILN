# Runtime Dispatch — Design

## Authority

Phase 3 of the KILN Master File v7.5. Rule KILN-E36 (the registry
is the hot path) applies at the runtime layer, not only the kernel
layer.

## The two layers

KILN has two dispatchers, one per layer.

| Layer | Location | Purpose |
|---|---|---|
| Kernel | `kiln-core::dispatch` | Per-matmul routing inside our own transformer. Frozen. Phase 5 may use it when hooking llama.cpp's graph. |
| Runtime | `kiln-runtime::runtime_dispatch` | Per-model-operation routing over the running backend. New in Phase 3. |

Same principle, different granularity. This document specifies the
runtime layer.

## Why the runtime layer is separate

`kiln-core::dispatch` carries `weight_bytes`, `quant`, `activations`,
`scratch` — the shapes our from-scratch kernels need. `kiln-runtime`
does not expose those. It exposes whole-model operations: load,
forward, reset, tokenize, logits, token_to_str. The two layers
cannot share an enum. They share a concept.

## RuntimeOp

Load creates a session. Unload destroys it. The others operate on a
live session.

- Load { path, n_ctx, n_batch, n_threads }
- Forward { tokens }
- Reset
- Tokenize { text }
- Logits
- TokenToStr { token }
- Unload
- NVocab
- EosToken

## RuntimeBackend trait

Four methods: name, supports, estimate, execute.

execute returns a RuntimeResult. The dispatcher does not inspect the
variant. The caller matches on what it asked for.

RuntimeResult variants: Unit, Logits(Vec<f32>), Tokens(Vec<i32>),
Str(String), Int(i32).

## RuntimeDispatcher

Holds a Vec of boxed RuntimeBackends. dispatch takes a RuntimeOp,
finds the first backend whose supports returns true, calls execute.

Phase 3 registers exactly one backend: llama_cpp. The choice rule is
first-that-supports. Phase 5 replaces it with a real cost model.

## LlamaCppBackend

Wraps `LlamaContext`. Owns the live session. Every RuntimeOp is a
thin call into the existing context.

- Load: constructs a LlamaContext via load_with and stores it.
- Forward: calls ctx.forward(tokens).
- Reset: calls ctx.reset().
- Tokenize: calls ctx.tokenize(text).
- Logits: returns the last computed logits. Phase 3 keeps a small
  cache so Logits after Forward is free.
- TokenToStr: calls ctx.token_to_str(token).
- Unload: drops the LlamaContext.
- NVocab: returns ctx.n_vocab().
- EosToken: returns ctx.eos_token().

LlamaContext itself is unchanged.

## Error model

RuntimeError variants: NoBackend, NoSession, Llama(LlamaError),
WrongReturn. From<LlamaError> exists.

WrongReturn fires if the caller asks for Logits but the result does
not match. Programming error, not runtime. It exists so the
dispatcher never silently returns the wrong shape.

## Changes to the CLI

Before: the CLI creates a LlamaContext and calls methods directly.

After: the CLI creates a RuntimeDispatcher, registers
LlamaCppBackend, and calls dispatch for every operation.

Same behavior. Different path. One vtable call per operation.

## What does not change

- LlamaContext and the C++ shim. Unchanged.
- Output. Byte-identical.
- Speed. Within 5%.
- kiln-core::dispatch. Frozen. Phase 5 may use it.

## Files

New:
- crates/kiln-runtime/src/runtime_dispatch.rs
- crates/kiln-runtime/src/backends/mod.rs
- crates/kiln-runtime/src/backends/llama_cpp.rs

Modified:
- crates/kiln-runtime/src/lib.rs
- crates/kiln-cli/src/main.rs

## Tests

Test 1: dispatcher routes Load. Fails with zero backends.
Test 2: Forward before Load returns NoSession.
Test 3: differential. hi there -> Hello! How can I help you today?
Test 4: speed within 5% of 0.1497 s.

## Gate 3

- kiln debug prints Paris.
- kiln chat turn 1 gives the same reply at 10.00 tok/s.
- Output byte-identical to Phase 2.3.
- Speed within 5%.
- Every model operation goes through the dispatcher.

## Anti-patterns

- Do not put runtime ops into kiln-core::dispatch.
- Do not make LlamaContext implement RuntimeBackend directly.
- Do not add a second backend in Phase 3.
- Do not change LlamaContext's public API.
- Do not change the shim.
