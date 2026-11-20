# KILN

**Run local language models with a runtime that adapts to the machine you already own.**

KILN is an open-source inference runtime for local language models. It includes a native desktop window on Linux, a loopback-only Ollama-compatible API, model discovery, hardware-aware setup, and experimental paths for compatible models that exceed available memory.

KILN is still under active development. Its large-model speed, quality, memory and storage targets have **not** been accepted on the Tier 0 test laptop. Experimental streaming and passing CI are not deployment certification.

## Desktop workspace

On Kali and Debian, the prebuilt package installs KILN’s desktop launcher and GTK/WebKit runtime. Run KILN from the app menu or:

    kiln dashboard

The app opens in its own window. It checks the machine, finds installed models, and may start a suitable candidate automatically. Chat, Models, Activity, and System details are served locally. KILN does not download model weights during install or send prompts to a cloud service.

The previews below show the approved interface direction. Model and hardware values are illustrative, not captured from a live run.

![KILN chat workspace design preview](docs/screenshots/kiln-chat-preview.png)

![KILN dashboard design preview](docs/screenshots/kiln-dashboard-preview.png)

### Install on Kali or Debian

Download the amd64 or arm64 .deb from [KILN Releases](https://github.com/andreipath26/KILN/releases), then install it:

    sudo apt install ./kiln-linux-ARCH-COMMIT.deb

Replace the example filename with the downloaded asset. Launch KILN from the app menu or run kiln dashboard. Main-branch packages are previews; use a versioned release for the stable update channel.

## Build from source

Requirements: Git, Rust/Cargo, CMake, a C/C++ toolchain, and the pinned llama.cpp submodule.

    git clone --recurse-submodules https://github.com/andreipath26/KILN.git
    cd KILN
    cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
    cmake --build vendor/llama.cpp/build --config Release -j
    cargo run -p kiln-cli -- --help

Run the dashboard from the checkout with:

    cargo run -p kiln-cli -- dashboard

On Linux, this opens the local UI in the GTK/WebKit window. The Debian package installs the desktop dependencies automatically. On macOS and Windows, dashboard opening still uses the system browser.

## Models and chat

KILN discovers local GGUF files and installed Ollama model manifests. When an installed Ollama tag resolves to one unambiguous weight file, KILN can use it without copying weights.

    cargo run -p kiln-cli -- list
    cargo run -p kiln-cli -- list --json
    cargo run -p kiln-cli -- run auto
    cargo run -p kiln-cli -- run /path/to/model.gguf --max-tokens 50
    cargo run -p kiln-cli -- run 'your-installed-ollama-tag'

To find or download a public Hugging Face GGUF, choose the repository and file explicitly:

    cargo run -p kiln-cli -- search "model family"
    cargo run -p kiln-cli -- pull OWNER/REPOSITORY --file path/in/repository/model.gguf

Downloads are integrity-checked and stored under KILN_MODELS_DIR or the user data directory. For numbered split files, add --all-shards to pull a complete set. KILN checks that all listed parts are present and verifies each installed file. Incomplete remote shard groups are not offered as complete downloads.

The dashboard assesses available memory and model compatibility at startup. It may attempt up to three suitable installed models and shows failures in the interface. This avoids routine command-line tuning; it does not guarantee that an architecture is supported or that a model will run at a particular speed.

## Measure a model

Use measure to capture repeatable model, machine, runtime, memory, thermal, cache, and log evidence for a local GGUF:

    cargo run -p kiln-cli -- measure /path/to/model.gguf \
      --out ./measurements/model-run --tokens 128 --runs 5 --threads 4

Each run writes a new output directory with a summary, command records, JSON reports, model hash and per-stage stdout/stderr. Failed stages keep their logs and do not erase earlier evidence. Linux runs sample process RSS, host and cgroup memory, CPU frequency, thermal zones, and available block-device counters. Add a reference GGUF and corpus to compare candidate and reference loss on the same text.

Some measurements are estimates or diagnostics. Linux page cache is shared and uncontrolled; block-device counters can include other processes; O_DIRECT replay is best-effort. Cache-policy replay is a simulation, not live inference. These measurements do not by themselves prove physical SSD traffic, total memory bounds, sustained throughput, or quality parity.

### One-command Tier 0 qualification

Copy [the example configuration](docs/tier0-qualification.example.json), set the required local model paths, and run all configured Phase 3.5, Phase 5, and Phase 6 captures:

    cargo run -p kiln-cli -- qualify \
      --config ./tier0-qualification.json \
      --out ./measurements/tier0-run

The command retains model hashes, machine metadata, per-command logs, reports, and failure details. It refuses to reuse an existing output directory. Review qualification.json and commands.json. A failed or incomplete capture returns a nonzero exit code.

The report deliberately leaves overall status inconclusive and deployment_ready false. A command cannot prove that it ran on the target laptop, replace manual review of quality results, or establish sustained thermal and isolated physical-storage limits.

## Experimental inference paths

### CPU MoE expert streaming

On Linux, KILN has an opt-in native streaming path for compatible fused-expert MoE GGUFs:

    cargo run -p kiln-cli -- plan /path/to/model.gguf --json
    cargo run -p kiln-cli -- stream /path/to/model.gguf --max-tokens 50
    cargo run -p kiln-cli -- stream /path/to/model.gguf --cache-mib 0

For supported layouts, selected expert slabs are read from the original GGUF rather than copying the model into a second file. The loader checks the expert layout and rejects unsupported or incomplete files. The dashboard can select this path for eligible oversized models. This is experimental inference, not proof of useful speed, bounded total RAM, or physical SSD throughput.

### LLaDA diffusion

Phase 6 includes a separate experimental diffusion command for supported LLaDA models:

    cargo run --release -p kiln-cli -- diffuse /path/to/llada.gguf \
      --prompt "Answer briefly." --max-tokens 32 --json

See the [Phase 6 guide](docs/phase6-diffusion.md) for the paired confidence-versus-CLAD evaluation and evidence capture.

## Local API and settings

The API listens on 127.0.0.1:11435 and provides an Ollama-compatible local endpoint. It remains available for terminal use, scripts, and compatible clients.

    kiln serve
    kiln dashboard --no-open

The first command starts the API without a desktop window. The second starts the API without opening the app. The UI and API bind to loopback; remote access is not included.

### Update a user-installed binary

Stable self-updates are limited to versioned GitHub releases:

    kiln update --check
    kiln update

KILN verifies the platform, manifest, asset size, and SHA-256 before replacing a binary in ~/.local/bin/kiln. The previous binary is retained for recovery. The updater does not install preview builds or overwrite system package-manager installations.

## Performance and deployment status

The research acceptance target is a 70B-class sparse MoE model with about 4B active parameters per token, at least 4 tokens per second, and quality within 1% of the pinned FP16 WikiText-2 reference on a Dell Latitude 7490 with an i7-8650U, 16 GB RAM, NVMe storage, and no discrete GPU.

**This target has not been demonstrated.** KILN’s experimental paths and CI smoke models do not establish the target throughput, quality, memory, storage, thermal reliability, or broad model compatibility. Treat published benchmark figures as historical measurements unless their model, hardware, command, and evidence are provided.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Quantization policy](docs/quantization-policy.md)
- [Phase 3.5 measurement guide](docs/phase35-measurement.md)
- [Phase 5 streaming guide](docs/phase5-streaming.md)
- [Phase 6 diffusion guide](docs/phase6-diffusion.md)
- [Predictor training and evaluation](docs/training-pipeline-design.md)
- [UMF and expert sidecar status](docs/umf-design.md)
- [Runtime loading and release design](docs/runtime-loading-design.md)
- [UI architecture](docs/ui-architecture.md)
- [Master roadmap and engineering record](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

KILN is licensed under MIT. See [LICENSE](LICENSE).
