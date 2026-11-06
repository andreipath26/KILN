# KILN

**A local-first inference runtime designed to run models larger than available memory, using the hardware already in your system.**

KILN is being built to coordinate model execution and memory across CPUs, GPUs, NPUs, APUs and integrated GPUs, RAM, NVMe storage, and accelerator memory. Its goal is a single runtime and model workflow that can adapt to different hardware configurations.

## The idea

Large models often exceed the memory available on a personal computer. Mixture-of-experts (MoE) models activate only part of their weights for each token, creating an opportunity to keep the full model on storage while bringing the experts in use into faster memory.

KILN aims to combine per-layer hardware dispatch, expert prediction and caching, and a tiered memory hierarchy to make that practical while preserving model quality. These are engineering goals; KILN has not yet demonstrated its full large-model performance target.

## Project status

KILN is an active engineering project and is not yet a stable, production-ready runtime. The codebase includes a Rust workspace, llama.cpp integration, runtime dispatch, GGUF model support, TQ1.0 ternary support, a CLI, an API, and benchmarking and quality-checking tools. Dense-path performance and end-to-end MoE streaming remain under development.

The initial acceptance machine is a Dell Latitude 7490 with 16 GB RAM and no discrete GPU. It is a minimum validation target, not a limit on the hardware KILN is designed to support.

The project’s mission target is to run a 70B-class MoE model with about 4B active parameters at 4 tokens per second while staying within 1% of an FP16 reference on WikiText-2. This is a target, not a result.

## Platform approach

KILN is intended to be platform agnostic. The runtime should use the CPU, GPU, NPU, APU or integrated GPU capabilities available on a system, with appropriate backend implementations and safe fallback behavior. Hardware-specific kernels and memory policies fit behind a common runtime contract; support and performance depend on the backend and device.

## Explore the project

- [Master roadmap and engineering record](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) — authoritative phases, acceptance criteria, research, decisions, and measured results.
- [Architecture overview](docs/architecture.md)
- [Quantization policy](docs/quantization-policy.md)
- [Design documents](docs/)
- [Source code](crates/)

To inspect the current development CLI:

```sh
cargo run -p kiln-cli -- --help
```

## Contributing

KILN is open source under the [MIT License](LICENSE). Before proposing a runtime or performance change, review the roadmap and relevant design document so implementation and acceptance criteria stay aligned.

Performance claims should include reproducible measurements for the model, hardware, backend, workload, and build used.

