//! GitHub release updater for user-scoped Linux installations.
//! Never downloads repository HEAD or executes an unverified remote script.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const API: &str = "https://api.github.com/repos/andreipath26/KILN/releases/latest";
const MAX_METADATA: u64 = 1024 * 1024;
const MAX_BINARY: u64 = 512 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct Asset { name: String, browser_download_url: String, size: u64 }
#[derive(Debug, Deserialize)]
struct Release { tag_name: String, draft: bool, prerelease: bool, assets: Vec<Asset> }
#[derive(Debug, Deserialize)]
struct Manifest {
    artifact_type: String,
    os: String,
    architecture: String,
    kiln_commit: String,
    binary_sha256: String,
}

fn client() -> Result<Client, String> {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .user_agent("KILN-release-updater")
        .build().map_err(|e| e.to_string())
}

fn bytes(client: &Client, url: &str, cap: u64) -> Result<Vec<u8>, String> {
    // Release metadata and binary MUST originate from the project's official GitHub endpoints.
    if url != API && !url.starts_with("https://github.com/andreipath26/KILN/releases/download/") {
        return Err("refusing an unexpected release download URL".into());
    }
    let mut response = client.get(url).send().map_err(|e| format!("GitHub request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("GitHub returned HTTP {} for update request", response.status()));
    }
    if response.content_length().is_some_and(|n| n > cap) {
        return Err("GitHub update asset exceeds download limit".into());
    }
    let mut out = Vec::new();
    response.take(cap + 1).read_to_end(&mut out).map_err(|e| e.to_string())?;
    if out.len() as u64 > cap { return Err("GitHub update asset exceeds download limit".into()); }
    Ok(out)
}

fn select<'a>(release: &'a Release, arch: &str) -> Result<(&'a Asset, &'a Asset), String> {
    if release.draft || release.prerelease { return Err("latest release is not stable".into()); }
    if !matches!(arch, "x86_64" | "aarch64") { return Err("unsupported Linux architecture".into()); }
    let prefix = format!("kiln-linux-{arch}-");
    let mut candidates = release.assets.iter().filter(|asset| {
        let Some(suffix) = asset.name.strip_prefix(&prefix) else { return false };
        suffix.len() == 16 && suffix.ends_with(".bin") &&
            suffix[..12].bytes().all(|c| c.is_ascii_hexdigit())
    });
    let binary = candidates.next().ok_or("release has no compatible standalone Linux binary")?;
    if candidates.next().is_some() { return Err("release contains ambiguous binary artifacts".into()); }
    let manifest_name = binary.name.trim_end_matches(".bin").to_owned() + ".json";
    let manifest = release.assets.iter().find(|a| a.name == manifest_name)
        .ok_or("release is missing the matching manifest")?;
    if binary.size == 0 || binary.size > MAX_BINARY || manifest.size == 0 || manifest.size > MAX_METADATA {
        return Err("release artifact size is invalid".into());
    }
    Ok((binary, manifest))
}

fn validate(manifest: &Manifest, arch: &str, binary: &Asset) -> Result<(), String> {
    if manifest.artifact_type != "single_executable_linux_release" ||
       manifest.os != "linux" || manifest.architecture != arch ||
       manifest.kiln_commit.len() != 40 ||
       !manifest.kiln_commit.bytes().all(|c| c.is_ascii_hexdigit()) ||
       manifest.binary_sha256.len() != 64 ||
       !manifest.binary_sha256.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid or mismatched release manifest".into());
    }
    let prefix = format!("kiln-linux-{arch}-");
    let short = binary.name.strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix(".bin")).ok_or("wrong binary name")?;
    if !manifest.kiln_commit.starts_with(short) {
        return Err("binary filename does not match the manifest commit".into());
    }
    Ok(())
}

fn installed_target() -> Result<PathBuf, String> {
    if !cfg!(target_os = "linux") { return Err("automatic updates are currently Linux-only".into()); }
    let exe = std::env::current_exe().map_err(|e| format!("cannot identify installed KILN: {e}"))?;
    let home = std::env::var_os("HOME").ok_or("HOME is missing")?;
    let expected = PathBuf::from(home).join(".local/bin/kiln");
    // Do not overwrite system package manager installations or an arbitrary executable.
    if exe != expected || !expected.is_file() {
        return Err("self-update supports ~/.local/bin/kiln only; other installs must be updated by their installer".into());
    }
    let metadata = fs::symlink_metadata(&expected).map_err(|e| e.to_string())?;
    if metadata.file_type().is_symlink() {
        return Err("refusing to replace a symlinked KILN binary".into());
    }
    Ok(expected)
}

fn staged_file(path: &Path, data: &[u8]) -> Result<(), String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o700);
    }
    let mut output = options.open(path).map_err(|e| e.to_string())?;
    output.write_all(data).and_then(|_| output.sync_all()).map_err(|e| e.to_string())
}

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) { let _ = fs::remove_file(&self.0); }
}

fn install(target: &Path, data: &[u8]) -> Result<(), String> {
    let dir = target.parent().ok_or("invalid KILN install path")?;
    let temporary = dir.join(format!(".kiln-update-{}", std::process::id()));
    let _cleanup = Cleanup(temporary.clone());
    staged_file(&temporary, data)?;
    let ok = Command::new(&temporary).arg("--help").output()
        .map_err(|e| format!("downloaded binary cannot launch: {e}"))?;
    if !ok.status.success() { return Err("new KILN failed its startup smoke test".into()); }
    let backup = dir.join("kiln.previous");
    if backup.exists() { return Err("kiln.previous already exists; move it before updating again".into()); }
    fs::rename(target, &backup).map_err(|e| format!("cannot preserve existing KILN: {e}"))?;
    if let Err(e) = fs::rename(&temporary, target) {
        let rollback = fs::rename(&backup, target);
        return Err(format!("installation failed: {e}; rollback: {rollback:?}"));
    }
    let _ = File::open(dir).and_then(|f| f.sync_all());
    Ok(())
}

/// Check GitHub's latest *published stable* release; if different from the
/// embedded build commit, verify and replace a user-scoped installed binary.
pub fn run(check_only: bool) -> Result<(), String> {
    if !cfg!(target_os = "linux") { return Err("kiln update currently supports Linux only".into()); }
    let arch = std::env::consts::ARCH;
    let client = client()?;
    let response = client.get(API).send().map_err(|e| format!("GitHub unavailable: {e}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        println!("No published stable KILN release is available yet. Installed binary unchanged.");
        return Ok(());
    }
    if !response.status().is_success() { return Err(format!("GitHub release API returned {}", response.status())); }
    if response.content_length().is_some_and(|n| n > MAX_METADATA) { return Err("oversized release metadata".into()); }
    let mut reader = response.take(MAX_METADATA + 1);
    let mut raw = Vec::new();
    reader.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    if raw.len() as u64 > MAX_METADATA { return Err("oversized release metadata".into()); }
    let release: Release = serde_json::from_slice(&raw).map_err(|e| format!("invalid GitHub release: {e}"))?;
    let (binary, manifest_asset) = select(&release, arch)?;
    let metadata = bytes(&client, &manifest_asset.browser_download_url, MAX_METADATA)?;
    let manifest: Manifest = serde_json::from_slice(&metadata).map_err(|e| format!("invalid manifest: {e}"))?;
    validate(&manifest, arch, binary)?;
    let current = option_env!("KILN_BUILD_COMMIT").unwrap_or("unknown");
    println!("KILN installed commit: {current}");
    println!("Latest stable release: {} ({})", release.tag_name, manifest.kiln_commit);
    if current == manifest.kiln_commit {
        println!("KILN is up to date.");
        return Ok(());
    }
    if check_only {
        println!("A different release is available. Run `kiln update` to install it.");
        return Ok(());
    }
    let target = installed_target()?;
    let payload = bytes(&client, &binary.browser_download_url, MAX_BINARY)?;
    if payload.len() as u64 != binary.size {
        return Err("downloaded binary length differs from GitHub release asset".into());
    }
    let actual = format!("{:x}", Sha256::digest(&payload));
    if actual != manifest.binary_sha256 {
        return Err("release binary SHA-256 does not match the manifest".into());
    }
    install(&target, &payload)?;
    println!("Updated KILN at {}. Previous binary: {}",
        target.display(), target.with_file_name("kiln.previous").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_or_ambiguous_assets() {
        let a = Asset { name: "kiln-linux-x86_64-123456789abc.bin".into(),
            browser_download_url: "https://github.com/andreipath26/KILN/releases/download/v1/test".into(), size: 10 };
        let m = Asset { name: "kiln-linux-x86_64-123456789abc.json".into(),
            browser_download_url: a.browser_download_url.clone(), size: 10 };
        let release = Release { tag_name: "v1".into(), draft: false, prerelease: false, assets: vec![a,m] };
        assert!(select(&release, "x86_64").is_ok());
        assert!(select(&release, "armv7").is_err());
        let mut duplicate = release;
        let same_name = duplicate.assets[0].name.clone();
        duplicate.assets.push(Asset { name: same_name,
            browser_download_url: String::new(), size: 10 });
        assert!(select(&duplicate, "x86_64").is_err());
    }
    #[test]
    fn refuses_wrong_manifest_commit() {
        let a = Asset { name: "kiln-linux-x86_64-123456789abc.bin".into(),
            browser_download_url: String::new(), size: 100 };
        let m = Manifest { artifact_type: "single_executable_linux_release".into(),
            os: "linux".into(), architecture: "x86_64".into(),
            kiln_commit: "a".repeat(40), binary_sha256: "b".repeat(64) };
        assert!(validate(&m,"x86_64",&a).is_err());
    }
}
