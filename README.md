# KILN

**Run models larger than your machine’s memory on the hardware you already own.**

KILN is an open-source local inference runtime in active development. It is designed to coordinate model execution and memory across CPU, GPU, NPU, APU/iGPU, RAM, NVMe storage, and accelerator memory. Its central challenge is making large sparse models practical on a wide range of machines without treating any one hardware tier as the whole product.

## What is in the repository

- A Rust workspace with a per-layer runtime dispatcher and a llama.cpp integration.
- GGUF model loading, a TQ1_0 ternary path, CLI and Ollama-compatible API.
- Benchmarking and quality-checking tools for controlled runtime experiments.
- Phase 5 components for validated GGUF expert-tensor maps, separate expert sidecars, byte-bounded cache admission, guarded asynchronous reads, a Linux io_uring reader, and a `cpu_moe_streamed` expert-byte dispatcher.

KILN is not yet a stable, general-purpose runtime. The Phase 5 sidecar, cache, I/O, and dispatcher components are primitives: stock llama.cpp inference does not yet call the streamed-expert operations. Dense-path performance and end-to-end MoE streaming remain under development.

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

The current development build statically links the vendored `kiln-llama` submodule. On Linux, with Rust, CMake and a C/C++ toolchain installed:

```sh
git clone --recurse-submodules https://github.com/andreipath26/KILN.git
cd KILN

cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build vendor/llama.cpp/build --config Release -j

cargo run -p kiln-cli -- --help
cargo run -p kiln-cli -- run /path/to/model.gguf --max-tokens 50
# Optional, experimental draftless n-gram speculative decoding:
cargo run -p kiln-cli -- run /path/to/model.gguf --spec-type ngram-mod
```

Find local GGUF files, search public Hugging Face GGUF repositories, or download a selected file:

```sh
cargo run -p kiln-cli -- list
cargo run -p kiln-cli -- list --path /path/to/models
cargo run -p kiln-cli -- run auto
cargo run -p kiln-cli -- search "granite 4.1"
cargo run -p kiln-cli -- pull REPOSITORY/NAME --file path/in/repository/model.gguf
```

`run auto` uses a local candidate only when exactly one readable GGUF is found; with zero or multiple candidates it explains how to search, pull or choose a path. Downloads go to `KILN_MODELS_DIR` or the user's data directory. KILN checks the Hugging Face SHA-256 and parses the downloaded GGUF before installing it. HTTPS downloads use KILN's Rust HTTP client and SHA-256 implementation. Local scanning checks the usual model, Hugging Face cache, and Ollama storage locations; `list` validates GGUF structure, but a successful scan does not guarantee that the active inference backend supports that model architecture.

Start the local API and open `http://127.0.0.1:11435/` to browse local GGUFs, search public Hugging Face GGUF repositories, download a selected file, activate a model, and chat. You can optionally select a startup model with `KILN_MODEL_PATH`:

```sh
cargo run -p kiln-cli -- serve
# Optional startup model:
KILN_MODEL_PATH=/path/to/model.gguf cargo run -p kiln-cli -- serve
```

The Advanced view reports current CPU/RAM defaults, inference behavior, and capability status. Accelerator placement, MoE streaming controls, and performance telemetry appear only when the runtime exposes them; these are not available yet. The API binds to loopback only and adds browser security headers; remote access is unsupported until KILN has an authenticated remote-access design.

The CLI and backend interfaces are still changing. Check command help for the current options.

## Documentation

- [UI architecture and continuation guide](docs/ui-architecture.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Architecture](docs/architecture.md)
- [Runtime dispatch design](docs/runtime-dispatch-design.md)
- [GGUF design](docs/gguf-design.md)
- [Quantization policy](docs/quantization-policy.md)
- [Decode-rate measurement contract](docs/phase35-decode-rate-contract.md)
- [Phase 3.5 benchmark capture suite](docs/phase35-capture-suite.md)
- [MoE predictor training and evaluation](docs/training-pipeline-design.md)
- [Master roadmap and engineering record](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md)

KILN is licensed under MIT. See [LICENSE](LICENSE).
