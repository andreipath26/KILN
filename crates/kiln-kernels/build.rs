//! Build script for kiln-kernels.
//!
//! Compiles the C kernels in cpp/kernels/ and links them into the crate.
//! No CMake. No external build orchestration. cargo build builds
//! everything.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let kernels_dir = workspace_root.join("cpp").join("kernels");

    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    let sources = ["tq1_0.c"];

    let mut build = cc::Build::new();
    build.include(&kernels_dir);
    build.warnings(true);
    build.flag_if_supported("-O3");
    build.flag_if_supported("-std=c11");

    // Enable AVX2 on x86_64 targets. The C code uses __AVX2__ to gate the
    // SIMD path. Without this flag, the AVX2 code is not compiled and the
    // differential test is meaningless.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86_64" || target_arch == "x86" {
        // Unconditional flag. The compiler supports -mavx2 (verified on the
        // build machine). flag_if_supported can silently drop the flag if
        // the probe build fails for an unrelated reason, which is what
        // happened here.
        build.flag("-mavx2");
        println!("cargo:rustc-cfg=kiln_avx2_enabled");
    }

    for src in sources {
        let path = kernels_dir.join(src);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(&path);
    }
    build.compile("kiln_kernels");
}
