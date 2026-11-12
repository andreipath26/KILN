# Phase 6 — LLaDA diffusion with CLAD

**Scope:** add a native KILN inference path for LLaDA masked diffusion using Cluster-Level Attention-Guided Parallel Decoding (CLAD). This is a separate decoding mode from autoregressive `kiln run` and Phase 5 MoE streaming.

**User target:** LLaDA-8B on the Tier 0 Dell Latitude 7490, with CLAD reaching 6 generated tokens/s on a fixed factual-prompt suite. The speed and quality target is not yet validated.

## Implementation

`kiln diffuse MODEL.gguf --prompt "..." --max-tokens 128 --steps 128 --block-length 32 --threshold 0.70 --json` runs the native LLaDA path. It applies the GGUF model's chat template, fills response positions with the model mask token, and repeatedly evaluates the full non-causal sequence. Each pass proposes the top token and its probability for every position. The C++ llama.cpp fork exposes the mean attention probabilities from LLaDA's final four layers, averaged over heads. KILN groups adjacent high-confidence proposals into spans and uses CLAD's mutual top-attention conflict matching to select spans that can be committed together.

The attention export is opt-in and graph-cache aware. It is disabled during ordinary inference and enabled by the KILN diffusion step. Models that do not expose the expected LLaDA attention map fail closed instead of silently switching to a different decoder.

The initial reference configuration is 128 generated positions, four blocks of 32, 128 total denoising steps, greedy proposals, and 0.70 confidence threshold. These are starting settings from the CLAD LLaDA evaluation recipe, not tuned Tier 0 parameters. The current CLI caps the complete prompt-plus-generation sequence at 512 tokens and returns an error when the requested sequence does not fit.

## Acceptance protocol

Run the same pinned factual prompts under a pinned LLaDA-8B-Instruct GGUF and tokenizer/template. Record the KILN and llama.cpp commits, model SHA-256, operating system/kernel, CPU, available RAM, thread count, quantization, context, prompt IDs, generation settings, output text, elapsed decode time, generated token count, and process peak RSS.

For speed, report the median and spread across repeated runs after a warm-up, with model load excluded and the decode interval used by `--json`. The user target is at least 6 generated tokens/s on the Dell Latitude 7490.

For quality, compare the CLAD answers against the same pinned vanilla LLaDA decoder on the exact same factual prompt set. Record exact/accepted-answer accuracy and review failures. Per KILN's accuracy-preservation principle, the Phase 6 quality gate is no more than a one percentage-point drop against that baseline. Do not report Phase 6 complete from speed alone.

Run the held-out test only after the implementation phase is complete, as requested. No target-machine acceptance result is present yet.

## Boundaries

- Phase 6 work is proceeding under the user's explicit instruction while Phase 5 hardware acceptance remains open. This does not close Phase 3.5 or Phase 5, change their targets, or certify the 70B MoE mission.
- `kiln diffuse` is the supported Phase 6 entry point. The autoregressive REST/UI path and other diffusion architectures are not silently redirected to CLAD.
- CLAD's published results and the original Python implementation are reference material, not KILN performance or quality evidence.
- The 6 tok/s result, factual accuracy comparison, model memory footprint, and sustained thermal behavior require the target machine and remain open.
