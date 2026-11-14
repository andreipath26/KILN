use std::path::PathBuf;
use std::process::{Command, Stdio};

/// `kiln measure` is the small native CLI front end to the repository's
/// evidence orchestrator. Keeping the suite in Python lets it preserve raw
/// subprocess logs and compose existing project probes without duplicating
/// their parsing and safety rules in the runtime.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: String,
    out: Option<PathBuf>,
    tokens: usize,
    runs: usize,
    threads: Option<u32>,
    reference: Option<String>,
    corpus: Option<PathBuf>,
    cache_mib: u32,
    min_tps: f64,
    max_rss_gib: f64,
) -> Result<(), String> {
    if tokens == 0 || runs == 0 || threads == Some(0) {
        return Err("tokens, runs, and an explicitly provided thread count must be positive".into());
    }
    if cache_mib > 512 {
        return Err("--cache-mib must be in 0..=512".into());
    }
    if !min_tps.is_finite() || min_tps < 0.0 || !max_rss_gib.is_finite() || max_rss_gib <= 0.0 {
        return Err("--min-tps must be finite and nonnegative; --max-rss-gib must be finite and positive".into());
    }
    if reference.is_some() != corpus.is_some() {
        return Err("--reference and --corpus must be supplied together for quality comparison".into());
    }

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("scripts/kiln_measure.py");
    if !script.is_file() {
        return Err(format!("measurement runner not found at {}; run from a KILN source build", script.display()));
    }
    let mut command = Command::new("python3");
    command.arg(&script).arg("--model").arg(model)
        .arg("--tokens").arg(tokens.to_string())
        .arg("--runs").arg(runs.to_string())
        .arg("--cache-mib").arg(cache_mib.to_string())
        .arg("--min-tps").arg(min_tps.to_string())
        .arg("--max-rss-gib").arg(max_rss_gib.to_string());
    if let Some(out) = out { command.arg("--out").arg(out); }
    if let Some(threads) = threads { command.arg("--threads").arg(threads.to_string()); }
    if let Some(reference) = reference { command.arg("--reference").arg(reference); }
    if let Some(corpus) = corpus { command.arg("--corpus").arg(corpus); }
    let status = command.stdin(Stdio::null()).status()
        .map_err(|error| format!("could not start python3: {error}"))?;
    if !status.success() {
        return Err(format!("measurement suite exited with {status}; review the run directory's logs and summary.json"));
    }
    Ok(())
}
