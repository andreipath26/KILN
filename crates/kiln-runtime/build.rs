//! Build script for kiln-runtime.
//! Compiles the KILN llama.cpp shim and unconditionally links against the
//! prebuilt static libraries under the vendored `vendor/llama.cpp/build`.
//! The dynamic loader design is not wired into this build script; see
//! docs/runtime-loading-design.md.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.parent().unwrap().parent().unwrap().join("vendor").join("llama.cpp");
    // A portable distribution MUST NOT accidentally pick up native-tuned
    // developer archives from vendor/llama.cpp/build. The release packager
    // supplies an isolated, explicitly non-native CMake build directory.
    let build_root = std::env::var_os("KILN_LLAMA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| llama_root.join("build"));
    let lib_dir = build_root.join("src");
    let ggml_lib_dir = build_root.join("ggml").join("src");
    let include_dir = llama_root.join("include");
    let ggml_include = llama_root.join("ggml").join("include");
    let shim_dir = PathBuf::from("../../cpp/llama_shim");

    assert!(lib_dir.exists(), "llama.cpp build/src not found at {}", lib_dir.display());
    assert!(ggml_lib_dir.exists(), "llama.cpp build/ggml/src not found at {}", ggml_lib_dir.display());
    assert!(lib_dir.join("libllama.a").exists(), "libllama.a not found");
    assert!(include_dir.join("llama.h").exists(), "llama.h not found");
    assert!(shim_dir.join("kiln_llama_shim.cpp").exists(), "shim .cpp not found");

    cc::Build::new()
        .cpp(true)
        .file(shim_dir.join("kiln_llama_shim.cpp"))
        .include(&include_dir)
        .include(&ggml_include)
        .include(&shim_dir)
        .warnings(true)
        .opt_level(3)
        .std("c++17")
        .compile("kiln_llama_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", ggml_lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=gomp");

    println!("cargo:rerun-if-env-changed=KILN_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-changed={}", lib_dir.join("libllama.a").display());
    println!("cargo:rerun-if-changed={}", ggml_lib_dir.join("libggml-cpu.a").display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shim_dir.join("kiln_llama_shim.cpp").display());
    println!("cargo:rerun-if-changed={}", shim_dir.join("kiln_llama_shim.h").display());
}
