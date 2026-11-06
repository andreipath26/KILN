# llama.cpp runtime loading: current implementation and shipping design

## Status

**Shipping design is not implemented.** This document records the code currently on main separately from the target release design. It does not make a legal determination about redistribution.

## Current implementation on main (2026-11-06)

- crates/kiln-runtime/Cargo.toml enables linked-llama by default. dynamic-llama enables the optional libloading dependency.
- crates/kiln-runtime/build.rs runs regardless of feature selection. It requires the built vendor/llama.cpp/build artifacts, compiles cpp/llama_shim/kiln_llama_shim.cpp, and links static llama, ggml, ggml-base, and ggml-cpu libraries plus stdc++ and gomp.
- crates/kiln-runtime/src/ffi.rs calls the C shim through linked extern "C" functions. LlamaCppBackend uses that FFI path.
- loader.rs provides platform library-name/path helpers and an optional Library wrapper. Those helpers are not the active LlamaCppBackend loading path. The current dynamic feature does not, by itself, remove the static build requirements.
- The source build therefore depends on a prebuilt vendored fork and the platform libraries above. The current path has not been qualified as a clean-machine release for Linux, Windows, macOS, or accelerator backends.

## Shipping design to implement

Choose and document one release strategy per supported platform:

1. **Runtime-loaded llama.cpp:** build KILN without static llama.cpp references; resolve a versioned, pinned library through a stable function table; load only a compatible library; report precise missing-library/symbol/version errors.
2. **Platform release package:** ship a reproducible KILN build with the selected pinned llama.cpp libraries and their required notices/dependencies, built separately for each supported target.

Do not mix these strategies accidentally. Keep KILN's public model/API contract stable and make CPU fallback explicit when an accelerator backend is unavailable. Define the supported OS, architecture, accelerator and llama.cpp revision matrix before producing release artifacts.

## Required release behavior

- A clean build must not depend on a developer-specific absolute path or a manually prebuilt artifact hidden in a worktree.
- Pin the llama.cpp fork commit/revision and record compiler, build flags, backend options and ABI expectations.
- Verify checksums/signatures for downloaded artifacts if the chosen design downloads libraries. Never download or execute unverified binaries. Make download and update behavior explicit to the user.
- Confirm exact license notices and distribution obligations for the fork, linked libraries, transitive dependencies and each release package in the separate licensing review.
- Fail closed on missing or incompatible libraries; never silently switch to a backend with different correctness or performance behavior.
- Test the actual release artifact on a clean machine with a small pinned model, then run the per-platform differential and fallback suite.

## Current Linux development build

The Linux development path builds the vendored llama.cpp submodule before KILN. From the repository root:

```sh
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build vendor/llama.cpp/build --config Release -j
cargo build -p kiln-cli
```

These commands describe the current linked development path only. They are not a supported installer or cross-platform release procedure.
