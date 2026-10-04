# Runtime Loading of llama.cpp — Design

## Legal position

llama.cpp is MIT. KILN does not redistribute it. The user downloads
it from the official source on first run. KILN credits llama.cpp and
links to its license. No bundling, no source-disclosure obligation.

Precedent: AILOFlow states it "does not redistribute llama.cpp
binaries: it fetches them at runtime from the official releases."

## Dev vs shipping

**Development:** compile-time link against
/home/andreipath/llama.cpp/build/bin/. Fast iteration.

**Shipping:** no compile-time link. At runtime KILN locates or
downloads llama.cpp and loads it via dlopen / LoadLibrary.

Same FFI surface. Only the loader differs. Selected by a Cargo
feature flag.

## Discovery order on first run

1. System-wide install detected (/usr/lib, /usr/local/lib,
   ~/.local/lib, /opt/homebrew, Program Files, LOCALAPPDATA).
2. KILN-managed install already present in the user-data directory.
3. Download from github.com/ggml-org/llama.cpp/releases for the
   user's platform. Verify SHA256 published with the release.
   Extract only the shared libraries, not the executables.
4. If download fails, print the URL and exit. Never fall back to a
   bundled copy.

## The dlopen abstraction

New module: kiln-runtime/src/loader.rs.

    pub struct Library { handle: *mut c_void, path: PathBuf }
    impl Library {
        pub fn open(path: &Path) -> Result<Self, LoadError>;
        pub unsafe fn symbol(&self, name: &str) -> Result<*mut c_void, LoadError>;
    }

    pub trait LlamaApi {
        unsafe fn model_load_from_file(&self, ...) -> *mut c_void;
        unsafe fn init_from_model(&self, ...) -> *mut c_void;
        // one function pointer per symbol used
    }

Two implementations:
- LinkedApi   -- compile-time linking, dev builds.
- DynamicApi  -- dlopen, shipping builds.

## Feature flags

    [features]
    default = ["linked-llama"]
    linked-llama = []
    dynamic-llama = ["dep:libloading"]

build.rs links llama when linked-llama is on, and does nothing when
dynamic-llama is on. libloading is optional, version 0.8.

## Symbols required for Phase 2

llama_backend_init, llama_backend_free,
llama_model_default_params, llama_context_default_params,
llama_model_load_from_file, llama_model_free,
llama_init_from_model, llama_free,
llama_model_n_vocab, llama_tokenize, llama_token_to_piece,
llama_decode, llama_get_logits,
llama_batch_init, llama_batch_free.

If any symbol is missing, KILN fails loudly with the llama.cpp
version it needs. No silent breakage.

## Version pin

One file: kiln-runtime/llama-pin.toml. Contains the release tag, the
per-platform download URL, and the SHA256. Bumping llama.cpp is one
file change. Do not chase HEAD.

## Anti-patterns

- Do not bundle llama.cpp in the release archive.
- Do not download executables, only shared libraries.
- Do not auto-download without asking the user.
- Do not skip SHA256 verification.
- Do not silently fall back to a bundled copy.

## Gate

cargo build -p kiln-runtime --release --no-default-features
--features dynamic-llama produces a binary that, given no local
llama.cpp, downloads it, verifies it, loads it, and runs the model.
