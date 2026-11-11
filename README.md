# KILN

**Run models larger than machine memory, at usable speed, on the hardware you already own.**

KILN is an open-source local inference runtime in active development, intended for LLMs and SLMs supported by its inference backends. The project aims to coordinate CPU, GPU, NPU/APU/iGPU, RAM, accelerator memory, and storage without requiring a particular model or a particular machine. Actual backend support and performance depend on the model and hardware; this is not yet a stable, general-purpose release.

## Current functionality

- Rust workspace, runtime dispatcher, GGUF inspection/loading, TQ1_0 experiments, CLI, and a loopback-only Ollama-compatible API with a local web UI.
- Discovery of existing GGUF files and installed Ollama manifests, including human-readable Ollama tags. Where a local tag resolves unambiguously, `kiln run <ollama-tag>` uses the existing weight blob without copying it.
- Public Hugging Face GGUF search and explicit model-file downloads, with integrity checks.
- `kiln plan MODEL [--json]`: read-only GGUF/system memory and CPU feasibility preflight. It is a diagnostic, **not** a guarantee that a model will load or run at a particular speed.
- **Experimental Linux CPU MoE streaming:** `kiln stream MODEL` resolves a compatible fused-expert GGUF, creates a temporary verified source index, and starts an isolated inference child using selected expert slabs read directly from the **original GGUF**. This is real opt-in graph compute, not just synthetic I/O simulation. It supports bounded native expert caching, optional parallel reads, and fail-closed stream audit checks; it does not require duplicating model weights.
- Phase 5 engineering tools for real router-ID traces, expert index/sidecar validation, asynchronous I/O, cache experiments, and model quality/throughput evidence.

**What this does not mean:** The streamed path is not automatically enabled for every model, architecture or operating system. Ordinary `kiln run` remains the default inference path; incompatible expert layouts are rejected for automatic streaming. The existing `cpu_moe_streamed` byte-dispatch components and native GGML selected-expert computation are separate mechanisms, and neither is a general hardware-deployment acceptance certificate.

## Get started (Linux development build)

Prerequisites: Git, Rust/Cargo, CMake, a C/C++ toolchain, and the pinned `kiln-llama` submodule. Clone with submodules and build the native static libraries:

```sh
git clone --recurse-submodules https://github.com/andreipath26/KILN.git
cd KILN

cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build vendor/llama.cpp/build --config Release -j

cargo run -p kiln-cli -- --help
```

Find and use an already installed local model (GGUF or an unambiguous installed Ollama tag), or explicitly search/download a public GGUF:

```sh
cargo run -p kiln-cli -- list
cargo run -p kiln-cli -- list --json
cargo run -p kiln-cli -- list --path /path/to/models
cargo run -p kiln-cli -- run auto
cargo run -p kiln-cli -- run /path/to/model.gguf --max-tokens 50
cargo run -p kiln-cli -- run 'your-installed-ollama-tag'
cargo run -p kiln-cli -- search "model family"
cargo run -p kiln-cli -- pull OWNER/REPOSITORY --file path/in/repository/model.gguf
```

`run auto` selects a model only when discovery returns a single usable, unambiguous candidate; otherwise choose an explicit path or tag. Discovery does not guarantee that the backend supports a model's architecture. `pull` requires an explicit GGUF file in the selected repository; KILN does **not** automatically download model weights during installation. Downloads use `KILN_MODELS_DIR` or the user's data directory and validate the retrieved file.

Start the local server and UI:

```sh
cargo run -p kiln-cli -- serve
# Optional startup model:
KILN_MODEL_PATH=/path/to/model.gguf cargo run -p kiln-cli -- serve
```

Open **http://127.0.0.1:11435/**. The UI supports local model inventory, public GGUF search/download, model activation and chat. The API is loopback-only; authenticated remote access and automatic heterogeneous hardware placement are not shipped.

## Reusing an active model

Selecting a GGUF that is **already active** through the local UI does not tear down and reload its native inference session. KILN compares canonical source paths and preserves the existing model, native expert hot cache and execution mode, avoiding unnecessary storage reads and startup delays. Explicitly requesting streaming for a model currently running in standard mode still performs a deliberate mode switch. This optimization does not introduce an automatic background model download or alter local weight files.

## Oversized models in the local web UI

When you select **Use model**, KILN estimates a conservative resident-memory budget using Linux `MemAvailable` and, where present, remaining cgroup memory headroom. It reserves room for the OS, inference context and other allocations; comparing a GGUF with **total installed RAM alone** is insufficient. On Linux, an oversized **compatible single-file fused MoE GGUF** is launched through the experimental native on-demand expert streaming engine, with the model kept in a separate worker process for chat and generation. The Advanced tab identifies the active inference mode. An explicit **Try streaming** option is available for smaller compatible MoE files as well. Streaming remains experimental and is not proof of usable speed, bounded physical RAM, or support for every GGUF architecture.

**Split-model activation fails closed:** the API checks that every numbered source shard is a readable regular file and uses checked arithmetic for the combined size before deciding whether to load weights in RAM or stream them. Missing/unreadable shards and size overflow produce an explicit error rather than an underestimated resident-memory budget. Existing weights are not modified.

**Split GGUFs** with conventional names such as `model-00001-of-00002.gguf` must have **all shards in the same directory**. KILN detects missing parts, shows which filenames are absent, and avoids attempting to activate an incomplete set. With all shards present, KILN selects shard 1 as the entrypoint and tries the supported loader with lazy routed-expert mmap for models exceeding RAM. The native external expert streaming index now also supports complete, compatible split-GGUF fused layouts through versioned per-shard source mappings. It remains opt-in experimental inference, and not all models or quantizations are supported. Streamed API sessions use bounded JSON-lines messages between the local server and its isolated native worker. Invalid or truncated frames fail closed, client disconnects do not desynchronize subsequent responses, and a worker that does not exit at model-switch shutdown is terminated after a bounded grace period. Do not delete or re-download an existing shard unless its integrity is in question.

A successful GGUF metadata scan is not evidence that its architecture or quantization is supported by the native inference fork. Incompatible layouts or failed memory allocations remain explicit load errors rather than silent fallback claims.

## Download complete split models and resume safely

For public Hugging Face models supplied as numbered GGUF shards, the web UI offers **Download complete model (N shards)** when the repository lists every part. It downloads each original GGUF file separately with size and SHA-256 validation. If a transfer is cancelled, already completed shards remain intact, and the partial in-progress file is removed. A later attempt verifies and reuses existing shards rather than overwriting or downloading them again. Incomplete remote shard groups are not advertised as complete models.

The CLI provides the same complete-file workflow:

```sh
kiln pull OWNER/REPOSITORY --file "model-00001-of-00002.gguf" --all-shards
```

Use a nested repository file path when appropriate. The command checks that all numbered parts and their trusted digests exist in the public repository before starting transfers, and it verifies each installed file. **This is download completeness, not inference compatibility**: some architectures and quantizers remain unsupported, and KILN still requires native multi-shard expert compute validation before calling large split-model streaming production-ready.

## Real-model streamed-session recovery gate

The Linux CI smoke test now keeps one actual native MoE model loaded while exercising two HTTP 400 paths: an over-context prompt that fits the raw IPC request bound, and a control-character prompt whose escaped JSONL frame exceeds the native worker limit. It then sends a short generation request to the same native session and verifies that the streaming mode remains active. The response and recovery evidence are archived after the full smoke sequence. Passing this gate verifies protocol/session recovery for a small test GGUF only; it is not a large-model Tier 0 speed, memory, SSD I/O, or WikiText-2 quality result.

## Repeatable real-model CI fixtures

The smoke workflow reuses verified GGUF files by hard link for alternate-model and Ollama-manifest checks. Restored caches may already contain these paths, so fixture setup compares inode identity and only repairs a missing or stale link; it does not copy or modify source model weights. This avoids failing an otherwise successful native integration run because a cache-restored test link already exists.

## Phase 5 CI evidence preservation

The GitHub Actions real-model evidence archive runs **after** the complete model, split-GGUF, native-worker, portable-release, and browser smoke sequence (including API shutdown). The archive step uses `always()` so failed tests still preserve whatever evidence was produced. A failed artifact upload remains a visible CI failure and must be investigated; an uploaded artifact is not proof that every test or Tier 0 performance target passed.

## GGUF tensor safety and oversized metadata

GGUF tensor-size inspection uses checked, row-oriented encoded-byte arithmetic. Known quantization formats with malformed row dimensions, arithmetic overflow, or out-of-file tensor ranges are rejected instead of being treated as valid but unreadable weights. Unknown quantization types remain visible to discovery and diagnostics, but even unknown tensor offsets must belong to the source file. Parsing rejects out-of-range metadata offsets and data-section alignment overflow. This protects model inspection and expert-source mapping; it does **not** make an unsupported model architecture runnable.


## Split GGUF Phase 5 source mapping

`kiln plan /path/to/model-00001-of-00002.gguf --json` validates **every** part of a split GGUF and reports the **combined** model size, per-shard tensor counts, and verified source byte ranges for recognized routed-expert tensors. It does not combine the original files or read all the model weights into RAM. Supported tensor types are checked for row alignment, byte-range overflow and source-file bounds, while unknown quantizers remain explicitly unsupported. Missing shards, inconsistent split metadata and duplicate tensor names fail closed.

The `kiln-io` multi-source expert reader can independently open and pin the original source files, read selected exact tensor ranges into caller-owned buffers, and fork independent positional readers for parallel access. It also resolves verified third-axis **fused expert slabs** across all model shards, allowing a specific (layer, expert, projection) range to be read without loading the whole tensor.

Benchmark the real source-file I/O with:

```sh
cargo run --release -p kiln-bench -- expert-io-split \
  --source-gguf /path/to/model-00001-of-00002.gguf \
  --requests 64 --max-read-mib 256 --json
```

This reports explicit read bytes, per-shard request counts, wall-clock throughput and maximum caller buffer. Linux page cache is uncontrolled; the figures **do not** represent physical SSD traffic, inference throughput or accuracy.
 **Experimental native multi-shard compute:** the pinned KILN llama.cpp CPU backend now consumes a versioned `KILN-MOE-STREAM-2` index, resolves every graph-selected expert slab to its original GGUF shard, and issues bounded positional reads from that source. `kiln stream` creates the temporary verified index automatically for complete, supported fused three-projection MoE layouts, while leaving model files unchanged. The UI may select this mode for eligible split models exceeding current memory headroom. Unknown quantizers, incomplete projections and unsupported architectures still fail closed. Support for a verified source layout is **not** a Tier 0 performance, quality or physical SSD-traffic acceptance result.

## Bound native model startup

For compatible oversized MoE models, the local UI launches an isolated native streaming worker. The initial model-loading handshake now has a bounded deadline rather than allowing an indefinite loading spinner. The default is **300 seconds**; operators may set `KILN_STREAM_STARTUP_TIMEOUT_SECS` (clamped to 30–900 seconds) when slow storage or large model metadata requires longer. On timeout, KILN terminates the dedicated native launcher process group, including nested workers, and reports an explicit activation error. Existing model files remain untouched, and failed model switches still attempt to restore the previous session. This does not guarantee that a model architecture is supported or that inference meets the Tier 0 throughput target.


## Memory-budgeted native MoE caching

When activating a compatible MoE model through the local UI, KILN now derives its native expert cache capacity from the **currently available resident-memory budget**, including Linux cgroup headroom. The experimental default is approximately 1 MiB of cache for each 64 MiB of estimated resident budget, capped at **256 MiB**, and it may be zero when memory is exhausted. If memory information is unavailable, KILN uses the previous 64 MiB fallback. The Advanced tab reports the chosen per-session native expert cache size.

This policy controls one *native expert hot cache*, not the process RSS, OS page cache, KV memory, working set of shared weights or SSD physical I/O. Cache sizing is a conservative heuristic, not evidence of sustained Tier 0 throughput or a guarantee of a model fitting in RAM. Expert cache sizes for explicit `kiln stream --cache-mib N` are still user-controlled.

## Experimental streamed MoE inference (Linux only)

For a **compatible fused-expert MoE GGUF**, use the explicit opt-in command:

```sh
cargo run -p kiln-cli -- plan /path/to/model.gguf --json
cargo run -p kiln-cli -- stream /path/to/model.gguf --max-tokens 50
# Or use an unambiguous installed Ollama tag:
cargo run -p kiln-cli -- stream 'your-installed-ollama-tag' --cache-mib 64 --parallel-reads
# For experimental zero-cache comparison:
cargo run -p kiln-cli -- stream /path/to/model.gguf --cache-mib 0
```

`stream` accepts a GGUF path, a resolvable installed model identifier, or `auto`. Options include `--cache-mib 0..512` (64 by default), `--parallel-reads`, `--max-tokens`, `--keep-context`, and `--lazy-moe-experts`. It verifies the recognized expert layout, uses a temporary index and audit file, then cleans them up. It is opt-in and fails on unsupported layouts rather than silently claiming successful streaming. `--lazy-moe-experts` alone is a separate mmap-loader experiment, **not** equivalent to native streamed compute.

The native streamed worker now preserves token output bytes without using lossy UTF-8 replacement on an incomplete final codepoint. Invalid UTF-8 fails explicitly, while valid multi-byte characters split across token pieces are assembled intact. Worker JSONL token limits reject invalid types instead of defaulting silently, and out-of-order response frames poison the session rather than being treated as recoverable prompt errors.

The API also verifies the **serialized** JSONL request size before sending it to the native worker. Quotes, backslashes, and control characters may expand during JSON escaping; oversized encoded frames are rejected locally with HTTP 400, preserving the loaded model instead of causing a worker protocol failure.

For the local API, an overlong streamed-model prompt returns HTTP 400 with a corrective context-length message without poisoning the active native session. Broken/truncated worker IPC remains HTTP 502 and requires a fresh worker. This distinction has framed JSONL regression coverage.

For long chats, the streamed worker caps generated tokens to the remaining model context instead of attempting a decode beyond `n_ctx`. A prompt whose tokenization itself exceeds the context is rejected with an actionable message; the loaded model remains available for the next shorter request.

The native audit distinguishes operand bytes delivered, explicit `pread` bytes and cache hits. Those counters are **not** proof of physical SSD reads, cold-storage throughput or lower end-to-end memory usage. Use pinned models, repeatable quality baselines and target-hardware measurements before comparing speed, RAM and accuracy. See the [Phase 5 streaming guide](docs/phase5-streaming.md) and [master roadmap](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) for test methodologies and acceptance criteria.

## Update an installed KILN (Linux)

For a user-scoped KILN binary installed at `~/.local/bin/kiln`:

```sh
kiln update --check   # Query the latest published stable GitHub release (no changes)
kiln update           # Download, verify and install a newer published stable release
```

The updater queries official [KILN GitHub Releases](https://github.com/andreipath26/KILN/releases). The release must include a matching `kiln-linux-ARCH-COMMIT.bin` executable and `kiln-linux-ARCH-COMMIT.json` manifest generated by the Linux release build. KILN checks architecture, pinned source commit, asset size and the manifest's SHA-256 digest, then launches the staged executable with `--help` before replacing the installed binary. The prior executable is preserved as a uniquely named `~/.local/bin/kiln.previous.*` backup for manual recovery; subsequent updates keep their own backups. An unsuccessful verification leaves the installed binary unchanged.

Self-update is intentionally restricted to `~/.local/bin/kiln`; use your package manager or installer for system-wide, development or symlink installations. It never runs a remote shell script, overwrites model files or fetches unreviewed GitHub main-branch builds. The update channel is *published stable releases*, not every repository commit. As of this implementation, the repository does not yet have a published GitHub release, so `kiln update` reports that none is available and makes no changes. Published GitHub metadata and same-release SHA-256 verification do not independently prove publisher authenticity; signed release provenance is future release-hardening work.

## Portable Linux release candidate

The repository includes scripts to **build** a portable Linux single-executable release candidate and **install** a previously built archive without overwriting an existing `kiln`. This is a developer/release-candidate workflow, not a published one-command installer.

```sh
# From a clean, submodule-initialized checkout with Rust, CMake and build tools:
bash scripts/build_linux_release.sh

# Use the exact archive emitted in dist/ (with its adjacent .json and .sha256):
bash scripts/install_linux_release.sh dist/kiln-linux-ARCH-COMMIT.tar.gz

# Ensure ~/.local/bin is on PATH, then:
kiln --help
kiln list
```

Replace `ARCH-COMMIT` with the actual artifact name. The installer verifies checksums, architecture and executable startup and refuses to overwrite an existing installation. The binary contains **no model weights** and still depends on compatible target-system libraries; portability and inference quality need real-machine validation.

## Performance and deployment status

KILN's research acceptance target includes a 70B-class sparse MoE workload (~4B active parameters per token) at **at least 4 tokens/s**, with quality within 1% of the specified FP16 WikiText-2 reference, on the Dell Latitude 7490 Tier 0 machine (i7-8650U, 16 GB RAM, NVMe, no discrete GPU). That machine is a *minimum test floor*, not a product hardware requirement or a model-size restriction. LLMs and SLMs of other sizes remain in scope.

**These targets have not been demonstrated or accepted.** In particular, an experimental streaming integration passing CI smoke tests does not prove sustained Tier 0 throughput, bounded total RAM/page cache, physical SSD traffic, thermal reliability, quality parity or broad model compatibility. Past benchmark figures are historical observations, not current performance promises; the [master roadmap](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md) retains their context.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Quantization policy](docs/quantization-policy.md)
- [Phase 3.5 measurement guide](docs/phase35-measurement.md)
- [Phase 5 streaming guide](docs/phase5-streaming.md)
- [Predictor training and evaluation](docs/training-pipeline-design.md)
- [UMF and expert sidecar status](docs/umf-design.md)
- [Runtime loading and release design](docs/runtime-loading-design.md)
- [UI architecture and continuation guide](docs/ui-architecture.md)
- [Master roadmap and engineering record](KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

KILN is licensed under MIT. See [LICENSE](LICENSE).
