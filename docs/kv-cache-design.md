# KV Cache — Design

## Why this exists

`Transformer::forward_tokens` currently recomputes K and V for every
position on every call. `ChatSession::generate_streaming` calls
`forward` once per generated token, each time passing the full history.

Cost per call: seq_len position-passes through 28 layers.
Cost for a reply of N tokens on a prompt of P tokens: sum_{s=P}^{P+N} s
layer-position-passes. For P=5, N=30, that is 585 passes.

Measured on Dell 7490: ~1.9 s per position-pass through all layers.
585 passes ≈ 18 minutes for a 30-token reply. This matches what the
user observed.

## What the cache does

Store K and V for every layer at every position already processed.
On a new call, only the new positions are projected through Q, K, V.
Q is fresh (needed for the new positions). K and V for old positions
are read from the cache. Attention for position i attends to the cache
plus the new positions. Only the last new position's hidden state is
projected to logits.

New cost: P prefill passes + N decode passes = P + N layer-position-
passes. For P=5, N=30, that is 35. ~17x reduction.

## Cache layout

Per layer, two contiguous f32 vectors:

  k: [max_cache_len, num_kv_heads * head_dim]
  v: [max_cache_len, num_kv_heads * head_dim]

For Qwen2.5-1.5B: num_kv_heads=2, head_dim=128, kv_dim=256.
Per position per layer: 256*4 + 256*4 = 2 KB.
28 layers: 56 KB per position.
max_cache_len=2048: 117 MB. Acceptable on 16 GB Tier 0.

max_cache_len is a new config field, default 2048, overridable via a
new --context flag on kiln chat. Overflow is a hard error, not a silent
truncation.

## Where the cache lives

Inside Transformer, not ChatSession. ChatSession continues to call
forward(&history) exactly as today. The trait does not change.

Transformer gains:
  cached_len: usize
  cached_tokens: Vec<u32>
  cache: Vec<LayerKv>   // one per layer
  max_cache_len: usize

## Algorithm

forward_tokens(&mut self, tokens: &[u32]) -> Vec<f32>:

  1. If tokens.len() >= cached_len AND tokens[..cached_len] ==
     cached_tokens, new_tokens = tokens[cached_len..].
     Else, reset_cache() and new_tokens = tokens.

  2. If cached_len + new_tokens.len() > max_cache_len, error.

  3. Embed new_tokens. h layout: [n_new, hidden].

  4. For each layer L:
       a. attn_norm on h -> normed. Shape [n_new, hidden].
       b. Q = proj(attn_q, q_dim, hidden, normed)   [n_new, q_dim]
          K = proj(attn_k, kv_dim, hidden, normed) [n_new, kv_dim]
          V = proj(attn_v, kv_dim, hidden, normed) [n_new, kv_dim]
       c. Add Q, K, V biases if present.
       d. RoPE on Q and K with ABSOLUTE positions.
          Position of new token i = cached_len + i.
          This requires extending apply_rope to take a position offset.
       e. Append K, V (post-bias, post-RoPE) to cache[L].
       f. For each query head hq, each new position i:
            abs_i = cached_len + i
            kvh = hq / group
            scores[j] = dot(Q[i, hq], cache_k[j, kvh]) * scale
                        for j in 0..=abs_i
            softmax(scores)
            out[i, hq] = sum_j probs[j] * cache_v[j, kvh]
       g. attn_proj = proj(attn_output, hidden, q_dim, attn_out)
       h. h += attn_proj    // residual on new positions only
       i. ffn_norm, gate, up, silu*up, down, residual (unchanged)
          but on n_new positions, not seq_len.

  5. Final RMSNorm on the LAST new position only.
  6. Output proj to logits. Return.
  7. cached_len += n_new. cached_tokens.extend(new_tokens).

## RoPE with offset

Current signature: apply_rope(&mut x, seq_len, num_heads, head_dim).
New signature: apply_rope(&mut x, offset, n, num_heads, head_dim).
pos = (offset + t) as f32, instead of t as f32.

This is the only correctness-critical change. RoPE on cached K is
already applied and never re-applied. RoPE on fresh Q must use the
absolute position of the query token. Getting this wrong produces
subtly wrong attention and is hard to spot from logits alone.

## Testing

Test 1 — bit-identical. Same prompt, same seed, same max_tokens.
Cached path vs non-cached path. Outputs must be identical. This is the
anchor. It is the test that proves the cache did not change behavior.

Test 2 — reuse. Call forward with [a,b,c,d,e]. Call forward with
[a,b,c,d,e,f]. Assert cached_len went 5 -> 6 and n_new was 1, not 6.
Requires a test-only accessor for cached_len.

Test 3 — reset. Call forward with [a,b,c]. Call forward with [x,y,z].
Assert cached_len went 3 -> 3 (reset then prefill), not 6.

Test 4 — overflow. Fill to max_cache_len, then one more token. Assert
error, not panic, not silent corruption.

## What this does NOT do

- Does not quantize the cache. That is Phase 3 (q8_0 KV cache).
- Does not offload the cache to SSD. Phase 3 (HERALD, KVDRIVE).
- Does not use a paged or block layout. Phase 2 if needed.
- Does not cache the attention output or the residual. Only K and V.
- Does not change the sampler, the tokenizer, or the CLI interface
  (except for the new --context flag, which is optional).

## Expected performance

30-token reply on 5-token prompt, Qwen2.5-1.5B, Dell 7490:
  Before: 585 layer-position-passes ≈ 18 minutes.
  After:  35 layer-position-passes ≈ 1 minute.
  Speedup: ~17x.

The next bottleneck after this is the scalar fused matmul. AVX2 will
give another 4-8x on the matmul itself. That is a separate design doc.

## Anti-patterns

- Do not apply RoPE to cached K again. RoPE is applied once, at write
  time, using the absolute position of that token.
- Do not use relative positions in RoPE. The cache is absolute.
- Do not silently truncate the cache. Overflow is an error.
- Do not store pre-RoPE K. The cache holds post-bias, post-RoPE K.
- Do not skip the bit-identical test. It is the whole point.
