use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Run the repository's complete Tier 0 evidence capture and retain all logs.
pub fn run(config: PathBuf, out: PathBuf) -> Result<(), String> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("scripts/run_tier0_qualification.py");
    if !script.is_file() {
        return Err(format!("qualification runner not found at {}; run from a KILN source build", script.display()));
    }
    let status = Command::new("python3")
        .arg(script)
        .arg("--config").arg(config)
        .arg("--out").arg(out)
        .stdin(Stdio::null())
        .status()
        .map_err(|error| format!("could not start python3: {error}"))?;
    if !status.success() {
        return Err(format!("qualification runner exited with {status}; review qualification.json and retained stage logs"));
    }
    Ok(())
}
