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
    for src in sources {
        let path = kernels_dir.join(src);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(&path);
    }
    build.compile("kiln_kernels");
}
