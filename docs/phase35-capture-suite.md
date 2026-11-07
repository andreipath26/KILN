# Phase 3.5 unified benchmark capture

Run the existing Phase 3.5 measurements as one reproducible capture after the model has been downloaded:

```sh
bash scripts/run_phase35_suite.sh /absolute/path/to/Qwen2.5-1.5B-Q4_K.gguf
```

For an Ollama comparison, first import the same local GGUF using the [Ollama GGUF import guide](https://github.com/ollama/ollama/blob/main/docs/import.mdx):

```sh
printf 'FROM /absolute/path/to/Qwen2.5-1.5B-Q4_K.gguf\n' > /tmp/kiln-phase35.Modelfile
ollama create kiln-phase35-qwen -f /tmp/kiln-phase35.Modelfile
KILN_OLLAMA_MODEL=kiln-phase35-qwen:latest \
  bash scripts/run_phase35_suite.sh /absolute/path/to/Qwen2.5-1.5B-Q4_K.gguf
```

Check `ollama show kiln-phase35-qwen:latest --modelfile` to confirm the imported source path before trusting the comparison. The API model digest is not the original GGUF SHA, so the capture cannot prove weight identity by itself.

The suite builds the release CLI once, runs all 15 thread/batch/microbatch dense configurations, measures Rust and native greedy sampling at a fixed configuration, runs three exact-parity n-gram draft widths, and compares second-turn KV reuse with full re-prefill. It then captures an isolated Linux `perf` profile so instrumentation does not contaminate throughput trials. If `KILN_OLLAMA_MODEL` names a local Ollama model imported from the same GGUF, the suite alternates measured KILN/Ollama runs at identical prompt, output limit, thread, batch and context settings, records raw completion speed through Ollama's local API, its version and model digest, and reports the rate ratio. It writes shared provenance, per-run JSON, process timing/RSS files, build output, the profile, and `summary.json` into the chosen output directory. Set `KILN_BENCH_TOKENS`, `KILN_BENCH_RUNS`, and `KILN_BENCH_WARMUP` to override the defaults (256, 5, and 1).

`summary.json` rejects missing matrix points, malformed rates, failed greedy/speculative token parity, or failed KV argmax parity. When Ollama is captured, it reports both rates, their ratio, and whether prompt, output limit, threads, batch, microbatch, and context match KILN's Rust-greedy run. It reports usable profile samples and the 27 decode-forward tok/s threshold separately from Tier 0 acceptance. If Linux `perf` is unavailable or blocked by kernel permissions, the suite records profiling as pending instead of failing the inference measurements. The Ollama report records its model digest and supplied GGUF SHA but cannot verify they correspond to the same weights; confirm this in `run-notes.txt` along with Dell identity, pinned model, power, thermal, and cache conditions. Do not use the whole-response sampled rate or the KV prefill result as decode throughput.

Run the capture on the Dell Latitude 7490 with the roadmap's specified dense model to close the performance evidence. A different 30 GB model is useful for stress testing but does not substitute for that named target or prove the MoE Phase 5 gate.
