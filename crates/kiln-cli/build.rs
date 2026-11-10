use std::process::Command;
fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout);
            let sha = sha.trim();
            if sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                println!("cargo:rustc-env=KILN_BUILD_COMMIT={sha}");
            }
        }
    }
}
