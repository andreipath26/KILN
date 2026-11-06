# KILN

**Run models larger than your machine’s memory on the hardware you already own.**

KILN is an open-source local inference runtime in active development. It is designed to coordinate model execution and memory across CPU, GPU, NPU, APU/iGPU, RAM, NVMe storage, and accelerator memory. Its central challenge is making large sparse models practical on a wide range of machines without treating any one hardware tier as the whole product.

## What is in the repository

- A Rust workspace with a per-layer runtime dispatcher and a llama.cpp integration.
- GGUF model loading, a TQ1_0 ternary path, CLI and Ollama-compatible API.
- Benchmarking and quality-checking tools for controlled runtime experiments.
- Design work for MoE expert prediction, caching and streaming across the memory hierarchy.

KILN is not yet a stable, general-purpose runtime. End-to-end MoE streaming and performance across the intended hardware tiers are still under development.

## Performance

The project’s mission target is a 70B-class MoE model with about 4B active parameters at **4 tokens/s or better**, with WikiText-2 accuracy within 1% of an FP16 reference. This is a target, not an achieved result.

| Workload | Recorded result | Status |
|---|---:|---|
| Qwen2.5-1.5B Q4_K, Dell Latitude 7490 | 10.57 tok/s | Historical dense-path measurement; rerun needed for a current comparison |
| Qwen3-30B-A3B MoE streaming | 3 tok/s target | End-to-end result not yet demonstrated |
| 70B-class MoE, about 4B active parameters | 4 tok/s target | Mission target; not yet demonstrated |

The historical dense comparison with Ollama used different prompt-cache behavior, so its numbers should not be read as a controlled head-to-head result. The [roadmap](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) contains the benchmark record and its qualifications.

## Hardware

The Dell Latitude 7490 (i7-8650U, 16 GB RAM, no discrete GPU) is KILN’s minimum acceptance machine. It is the test floor, not the limit of the project. The runtime is intended for heterogeneous systems too; backend support and measured performance vary by device and are being developed.

## Build and run

The current development build links against the vendored `kiln-llama` submodule. On Linux, with Rust, CMake and a C/C++ toolchain installed:

```sh
git clone --recurse-submodules https://github.com/andreipath26/KILN.git
cd KILN

cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build vendor/llama.cpp/build --config Release -j

cargo run -p kiln-cli -- --help
cargo run -p kiln-cli -- run /path/to/model.gguf --max-tokens 50
```

To start the local API server, set `KILN_MODEL_PATH` to a readable GGUF file:

```sh
KILN_MODEL_PATH=/path/to/model.gguf cargo run -p kiln-cli -- serve
```

The CLI and backend interfaces are still changing. Check command help for the current options.

## Documentation

- [Architecture](docs/architecture.md)
- [Runtime dispatch design](docs/runtime-dispatch-design.md)
- [GGUF design](docs/gguf-design.md)
- [Quantization policy](docs/quantization-policy.md)
- [Decode-rate measurement contract](docs/phase35-decode-rate-contract.md)
- [MoE predictor training and evaluation](docs/training-pipeline-design.md)
- [Master roadmap and engineering record](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md)

KILN is licensed under MIT. See [LICENSE](LICENSE).
