# Dispatch Integration — Design

## Authority

This document implements Phase 1.5 of the KILN Master File v5.0.
It is governed by Rule KILN-E36 (the registry is the hot path).

## Purpose

Today, `transformer.rs` calls `proj_seq`, `dot_packed`,
`q4k_matmul_scalar`, and `q6k_matmul_scalar` directly. Every matmul
is a hardcoded choice between two kernels, selected by a length
check on the weight bytes.

This is the central architectural gap. Every future addition —
ternary, MoE streaming, diffusion, iGPU, NPU, SSD tier, thermal
adaptation — must be a registration into a dispatcher, not a new
branch in `dot_packed`. Without this seam, KILN converges to
llama.cpp by accumulation.

Phase 1.5 builds the seam. It changes nothing about output. It
changes nothing about speed beyond 5 percent. It changes the shape
of the code so that everything after it is additive.

## Non-goals

- Not adding a second backend. Exactly one backend is registered at
  the end of Phase 1.5: the current CPU scalar path.
- Not optimizing the dispatcher. It is called per layer per token.
  A trait object vtable is fine. Profile later.
- Not changing output. Same prompt, same seed, same tokens, same
  replies.
- Not changing the public API. CLI, REST server, and public
  transformer methods keep their signatures.

## The abstraction

Three new types. One new module.

### Operation

The Operation enum describes what needs to be done. It does not say
how. It has five variants: Matmul, Attention, RmsNorm, Embedding,
Sample. Matmul carries the tensor id, the quantization kind, the
output feature count, and the input feature count.

TensorId is the identity of a weight tensor in the loaded model.
Already defined in kiln-mem. Reused here.

QuantKind is an enum of the quantization formats we understand:
F32, F16, Q4_K, Q6_K, TQ1_0. It exists in kiln-models::quantization.
Reused here.

### Backend

A trait with four methods: name, supports, estimate, execute.

name returns a short string. supports answers whether this backend
can perform this operation. estimate returns a cost in nanoseconds
or None. execute runs it.

A backend is a registration. It declares what it supports and what
it costs. The dispatcher never asks which kernel. It asks the
registry which backends support this, and the scheduler which of
those is cheapest.

### ExecContext

A borrowed mutable struct passed through dispatch. Holds references
to the weight table, the quantization table, a scratch buffer pool,
the output slice, the activation slice, and the sequence position.
Allocates nothing in the hot path except via Scratch.

## The dispatcher

Dispatcher holds a reference to the registry, a list of boxed
backends, and a small cache of last choices.

dispatch takes an Operation and an ExecContext. It filters the
registered backends by supports, picks the first candidate, calls
execute. In Phase 1.5 there is exactly one candidate for every
operation, so the choice is trivial. Phase 4 replaces the choice
rule with a real cost model.

## The single backend for Phase 1.5

CpuScalarBackend. name is cpu_scalar. supports returns true for
every Operation variant. estimate returns Some(0). execute matches
on the Operation variant and calls the existing kernels:

- Matmul: q4k_matmul_scalar, q6k_matmul_scalar, or the TQ1.0
  matmul_scalar, or inline F32/F16 dot products. This match is the
  current dot_packed logic, moved out of transformer.rs and into the
  backend. This is the last place the which-kernel decision lives.

- RmsNorm: the current rms_norm function, moved here.

- Attention: not moved in Phase 1.5. The transformer keeps its
  inline attention loop. The backend exposes a stub that returns an
  error if called. Phase 3 replaces it.

- Embedding: the current embed_token logic, moved here.

- Sample: not moved in Phase 1.5. The sampler stays in chat.rs.

## Changes to transformer.rs

Every call of the form
    self.proj_seq(&layer.attn_q, q_dim, hidden, &normed, n_new)
becomes a call of the form
    self.dispatcher.matmul_seq(&layer.attn_q_id, q_dim, hidden,
                               &normed, n_new, ctx)?

The matmul_seq helper is a small wrapper on the dispatcher that
loops over the sequence and calls dispatch once per batch.

layer.attn_q_id is a TensorId assigned at weight-load time. The
LayerWeights struct gains a TensorId for every tensor.

ctx is built once per forward pass and reused.

## What does NOT change

- The KV cache. Still in transformer.rs. Still bit-identical.
- The RoPE, the attention math, the softmax, the residual, the FFN.
  These stay inline in transformer.rs for Phase 1.5. Phase 3 fuses
  them.
- The sampler. Still in chat.rs.
- The tokenizer. Still in kiln-models.
- The CLI, REST API, chat loop. Public signatures unchanged.
- The output. Byte-identical.

## Tests

Test 1 — differential. On Qwen2.5-1.5B, hi there, greedy, seed 42,
max 9 tokens: the reply must equal the pre-seam output
Hello! How can I help you today?

Test 2 — registry coverage. Every Operation variant has at least
one registered backend that returns supports == true.

Test 3 — no bypass. Grep transformer.rs for direct calls to
proj_seq, dot_packed, q4k_matmul_scalar, q6k_matmul_scalar. Zero
occurrences outside the backend implementation file.

Test 4 — speed. Wall-clock time for the same prompt within 5
percent of the pre-seam time.

## Files touched

New:
- crates/kiln-core/src/dispatch.rs
- crates/kiln-core/src/backends/mod.rs
- crates/kiln-core/src/backends/cpu_scalar.rs
- docs/dispatch-integration-design.md (this document)

Modified:
- crates/kiln-core/src/lib.rs
- crates/kiln-core/src/transformer.rs
- crates/kiln-core/src/transformer_weights.rs
- crates/kiln-hal/src/registry.rs (only if the existing trait needs
  a lookup method)

## Sequencing

1. Write dispatch.rs with the types and a stub Dispatcher.
2. Write cpu_scalar.rs with the CpuScalarBackend, copying logic
   from the current dot_packed and proj_seq.
3. Write backends/mod.rs.
4. Register the modules in lib.rs.
5. Add TensorId to LayerWeights.
6. Replace calls in transformer.rs one at a time. Build after each.
7. Run the differential test. Output must equal pre-seam output.
8. Run the bypass grep. Must be zero.
9. Run the speed test. Within 5 percent.
10. Update README, CHANGELOG, roadmap. Commit. Push.

## Anti-patterns

- Do not add a second backend. The seam is the point.
- Do not special-case any operation to bypass the dispatcher.
- Do not change output. If the differential test fails, the seam
  is wrong, not the output.
- Do not add inline attributes to help the compiler.
- Do not build a real cost model. The stub is intentional.
- Do not touch the KV cache, the sampler, or the tokenizer.

## Gate 1.5

The output of kiln chat on Qwen2.5-1.5B, hi there, greedy, seed 42,
max 9 tokens is exactly Hello! How can I help you today?

The reply is byte-identical to commit c75225d.

The wall-clock time is within 5 percent of commit c75225d.

There is exactly one backend registered.

There are zero direct kernel calls in transformer.rs.
