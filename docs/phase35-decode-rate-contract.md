# Phase 3.5 generation speed accounting

The benchmark measures three **different** rates. Do not compare them interchangeably:

- `decode_tokens_per_second` (legacy name, retained for compatibility): output samples divided by the time in the sampling/output loop. The first generated token is sampled from the prefill logits; it is not itself the result of a decode-forward step. In a short response this can materially inflate apparent model decode speed.
- `decode_forward_tokens_per_second`: only the number of actual one-token `RuntimeOp::Forward` calls performed after prefill divided by their accumulated wall time. When the model terminates immediately, or `--tokens 1`, there are zero post-prefill forward passes; this metric is **0**, not a model benchmark of one free token. This is the closest available per-forward measure, but excludes sampler and Rust-side conversion overhead.
- `response_tokens_per_second_including_prefill`: output tokens divided by measured prompt prefill plus output-loop time (excluding model loading and prompt tokenization). Useful for request-time comparison, but **not** a whole CLI-session wall-clock or throughput benchmark.

For each run, the JSON includes `decode_forward_steps`, `decode_forward_ms`, and all three rates. The statistics object reports their distribution across repetitions. Zero-forward short generations are not enough to assess sustained decode performance: use sufficiently long prompts and generation lengths, record EOS termination, model hash, active context, CPU threads/batch, power/thermal state and repeated measurements. Keep identical protocols for Ollama comparisons.

**No performance gain is claimed by this change.** It corrects measurement definitions needed to determine whether the Phase 3.5 dense-speed gate is met.
