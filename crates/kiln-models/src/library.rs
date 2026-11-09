//! Local GGUF discovery and Hugging Face GGUF search/download helpers.

use crate::gguf::GgufFile;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};

const MAX_SCAN_DEPTH: usize = 12;
const MAX_SCAN_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, Serialize)]
pub struct LocalModel {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub source: String,
    pub valid_gguf: bool,
    /// All installed Ollama tags that refer to the same physical GGUF blob.
    /// A single blob remains one model for auto-selection and memory accounting.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemoteArtifact {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemoteModel {
    pub repository: String,
    pub downloads: u64,
    pub license: String,
    pub gated: bool,
    pub artifacts: Vec<RemoteArtifact>,
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn standard_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut add = |value: Option<std::ffi::OsString>| {
        if let Some(value) = value {
            let path = PathBuf::from(value);
            if !roots.contains(&path) {
                roots.push(path);
            }
        }
    };
    add(std::env::var_os("KILN_MODELS_DIR"));
    add(std::env::var_os("HF_HOME").map(|p| PathBuf::from(p).join("hub").into_os_string()));
    add(std::env::var_os("XDG_CACHE_HOME").map(|p| PathBuf::from(p).join("huggingface/hub").into_os_string()));
    add(std::env::var_os("OLLAMA_MODELS"));
    #[cfg(target_os = "windows")]
    add(std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("KILN/models").into_os_string()));
    if let Some(home) = home_dir() {
        for path in [
            home.join("Models"), home.join("models"),
            home.join(".cache/huggingface/hub"),
            home.join(".ollama/models"),
            home.join(".local/share/kiln/models"),
        ] {
            if !roots.contains(&path) { roots.push(path); }
        }
    }
    roots
}

fn inspect_gguf(path: &Path) -> bool {
    GgufFile::open(path).is_ok()
}

fn has_gguf_magic(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut file| file.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// Human-readable canonical Hugging Face cache identity for a local
/// snapshot GGUF symlink. This is a path-to-name mapping only: the original
/// source is always canonicalized and validated as GGUF before use.
fn huggingface_snapshot_name(path: &Path) -> Option<String> {
    let parts = path.components().filter_map(|c| match c {
        std::path::Component::Normal(s) => s.to_str(),
        _ => None,
    }).collect::<Vec<_>>();
    for (index, part) in parts.iter().enumerate() {
        let Some(repo) = part.strip_prefix("models--") else { continue };
        if index + 3 >= parts.len() || parts[index + 1] != "snapshots" {
            continue;
        }
        let fields: Vec<_> = repo.split("--").collect();
        if fields.len() != 2 || fields.iter().any(|x| x.is_empty()) {
            continue;
        }
        let revision = parts[index + 2];
        if revision.is_empty() || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue;
        }
        let filename = parts[index + 3..].join("/");
        if !filename.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        if fields.iter().any(|x| !x.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))) {
            continue;
        }
        return Some(format!("{}/{}/{}", fields[0], fields[1], filename));
    }
    None
}

fn collect_files(root: &Path, depth: usize, visited: &mut usize, seen: &mut HashSet<PathBuf>, aliases: &mut std::collections::HashMap<PathBuf, Vec<String>>, out: &mut Vec<PathBuf>) {
    if depth > MAX_SCAN_DEPTH || *visited >= MAX_SCAN_ENTRIES { return; }
    let Ok(entries) = fs::read_dir(root) else { return; };
    for entry in entries.flatten() {
        if *visited >= MAX_SCAN_ENTRIES { break; }
        *visited += 1;
        let path = entry.path();
        let Ok(meta) = fs::metadata(&path) else { continue; };
        if meta.is_dir() {
            // Do not follow directory symlinks; this prevents loops and duplicate full-tree scans.
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                collect_files(&path, depth + 1, visited, seen, aliases, out);
            }
        } else if meta.is_file() {
            let looks_like_gguf = path.extension().and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false);
            // Ollama stores GGUF weights as extensionless content-addressed blobs.
            if looks_like_gguf || has_gguf_magic(&path) {
                let hf_name = huggingface_snapshot_name(&path);
                let canonical = fs::canonicalize(&path).unwrap_or(path);
                if let Some(name) = hf_name {
                    let names = aliases.entry(canonical.clone()).or_default();
                    if !names.contains(&name) { names.push(name); }
                }
                if seen.insert(canonical.clone()) { out.push(canonical); }
            }
        }
    }
}

/// Resolve Ollama's registry manifests into familiar names without copying
/// model blobs or invoking the Ollama daemon. All paths stay inside the
/// configured models tree; malformed or oversized manifests are ignored.
fn ollama_names_for_roots(roots: &[PathBuf]) -> std::collections::HashMap<PathBuf, Vec<String>> {
    const MAX_MANIFESTS: usize = 10_000;
    const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
    let mut result: std::collections::HashMap<PathBuf, Vec<String>> =
        std::collections::HashMap::new();
    let mut scanned = 0usize;
    for root in roots {
        let manifests = root.join("manifests");
        let blobs = root.join("blobs");
        if !manifests.is_dir() || !blobs.is_dir() {
            continue;
        }
        let Ok(canonical_blobs) = fs::canonicalize(&blobs) else { continue };
        let mut dirs = vec![(manifests, 0usize)];
        while let Some((dir, depth)) = dirs.pop() {
            if scanned >= MAX_MANIFESTS { break; }
            let Ok(items) = fs::read_dir(&dir) else { continue };
            for entry in items.flatten() {
                if scanned >= MAX_MANIFESTS { break; }
                let Ok(file_type) = entry.file_type() else { continue };
                let path = entry.path();
                if file_type.is_dir() {
                    if depth < 4 { dirs.push((path, depth + 1)); }
                    continue;
                }
                if !file_type.is_file() || depth != 3 { continue; }
                scanned += 1;
                let Ok(relative) = path.strip_prefix(root.join("manifests")) else { continue };
                let pieces: Vec<_> = relative.components().filter_map(|c| match c {
                    std::path::Component::Normal(name) => name.to_str(),
                    _ => None,
                }).collect();
                if pieces.len() != 4 { continue; }
                let [registry, namespace, model, tag] =
                    [pieces[0], pieces[1], pieces[2], pieces[3]];
                if ![registry, namespace, model, tag].iter().all(|part| {
                    !part.is_empty() && !part.bytes().any(|b| b.is_ascii_control())
                }) { continue; }
                let Ok(metadata) = entry.metadata() else { continue };
                if metadata.len() == 0 || metadata.len() > MAX_MANIFEST_BYTES { continue; }
                let Ok(bytes) = fs::read(&path) else { continue };
                let Ok(manifest) = serde_json::from_slice::<Value>(&bytes) else { continue };
                let Some(layers) = manifest.get("layers").and_then(Value::as_array) else { continue };
                let alias = if registry == "registry.ollama.ai" && namespace == "library" {
                    format!("{model}:{tag}")
                } else if registry == "registry.ollama.ai" {
                    format!("{namespace}/{model}:{tag}")
                } else {
                    format!("{registry}/{namespace}/{model}:{tag}")
                };
                for layer in layers {
                    if layer.get("mediaType").and_then(Value::as_str)
                        != Some("application/vnd.ollama.image.model") {
                        continue;
                    }
                    let Some(digest) = layer.get("digest").and_then(Value::as_str)
                        .and_then(|v| v.strip_prefix("sha256:")) else { continue };
                    if digest.len() != 64 ||
                        !digest.bytes().all(|b| b.is_ascii_hexdigit()) { continue; }
                    let blob = canonical_blobs.join(format!("sha256-{}", digest.to_ascii_lowercase()));
                    let Ok(canonical_blob) = fs::canonicalize(&blob) else { continue };
                    if !canonical_blob.starts_with(&canonical_blobs) ||
                        !canonical_blob.is_file() { continue; }
                    let names = result.entry(canonical_blob).or_default();
                    if !names.contains(&alias) { names.push(alias.clone()); }
                }
            }
        }
    }
    for aliases in result.values_mut() { aliases.sort(); }
    result
}

pub fn discover_models(extra_roots: &[PathBuf]) -> Vec<LocalModel> {
    let mut roots = standard_roots();
    roots.extend(extra_roots.iter().cloned());
    let mut ollama_roots = Vec::new();
    if let Some(path) = std::env::var_os("OLLAMA_MODELS") {
        ollama_roots.push(PathBuf::from(path));
    }
    if let Some(home) = home_dir() {
        ollama_roots.push(home.join(".ollama/models"));
    }
    // Extra model roots may themselves be Ollama storage roots.
    ollama_roots.extend(extra_roots.iter().cloned());
    let ollama_names = ollama_names_for_roots(&ollama_roots);
    let mut files = Vec::new();
    let mut hf_aliases = std::collections::HashMap::<PathBuf, Vec<String>>::new();
    let mut seen = HashSet::new();
    let mut unique_roots = HashSet::new();
    let mut visited = 0usize;
    for root in roots {
        if root.is_dir() {
            let canonical = fs::canonicalize(&root).unwrap_or(root);
            if unique_roots.insert(canonical.clone()) {
                collect_files(&canonical, 0, &mut visited, &mut seen, &mut hf_aliases, &mut files);
            }
        } else if root.is_file() {
            let canonical = fs::canonicalize(&root).unwrap_or(root);
            if seen.insert(canonical.clone()) { files.push(canonical); }
        }
    }

    let mut models = Vec::new();
    for path in files {
        if !inspect_gguf(&path) { continue; }
        let Ok(metadata) = fs::metadata(&path) else { continue; };
        let mut aliases = ollama_names.get(&path).cloned().unwrap_or_default();
        if let Some(hf_names) = hf_aliases.get(&path) {
            for alias in hf_names {
                if !aliases.contains(alias) { aliases.push(alias.clone()); }
            }
        }
        aliases.sort();
        let name = ollama_names.get(&path).and_then(|names| names.first())
            .or_else(|| hf_aliases.get(&path).and_then(|names| names.first()))
            .cloned()
            .unwrap_or_else(|| path.file_name().and_then(|x| x.to_str())
                .unwrap_or("model.gguf").to_owned());
        models.push(LocalModel {
            name,
            path: path.display().to_string(),
            size_bytes: metadata.len(),
            source: if ollama_names.contains_key(&path) { "Ollama manifest".into() }
                else if hf_aliases.contains_key(&path) { "Hugging Face snapshot".into() }
                else { root_label(&path) },
            valid_gguf: true,
            aliases,
        });
    }
    models.sort_by(|a,b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then(a.path.cmp(&b.path)));
    models
}

pub fn resolve_auto_model() -> Result<PathBuf, String> {
    let models = discover_models(&[]);
    match models.as_slice() {
        [model] => Ok(PathBuf::from(&model.path)),
        [] => Err("no readable local GGUF was found; use `kiln search <query>` and `kiln pull <owner/repo> --file <file.gguf>`, or pass a model path".into()),
        many => {
            let choices = many.iter().map(|model| format!("  {} ({:.2} GiB)\n    {}", model.name,
                model.size_bytes as f64 / 1_073_741_824.0, model.path)).collect::<Vec<_>>().join("\n");
            Err(format!("found {} local GGUF candidates; choose one with `kiln run <path>`:\n{}", many.len(), choices))
        }
    }
}

fn root_label(path: &Path) -> String {
    let text = path.to_string_lossy();
    if text.contains("huggingface") { "Hugging Face cache".into() }
    else if text.contains(".ollama") { "Ollama storage".into() }
    else { "local GGUF".into() }
}

fn url_encode(value: &str) -> String {
    let mut out = String::new();
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn get_https_text(url: &str) -> Result<String, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .redirect(https_redirect_policy())
        .build().map_err(|e| format!("could not initialize HTTPS client: {e}"))?;
    let response = client.get(url).send().map_err(|e| format!("Hugging Face request failed: {e}"))?
        .error_for_status().map_err(|e| format!("Hugging Face request failed: {e}"))?;
    response.text().map_err(|e| format!("Hugging Face returned invalid response text: {e}"))
}

fn https_redirect_policy() -> Policy {
    Policy::custom(|attempt| {
        if attempt.url().scheme() == "https" && attempt.previous().len() < 10 {
            attempt.follow()
        } else {
            attempt.stop()
        }
    })
}

fn sha256_from_metadata(file: &Value) -> Option<String> {
    file.pointer("/lfs/sha256")
        .or_else(|| file.pointer("/lfs/oid"))
        .or_else(|| file.get("sha256"))
        .and_then(Value::as_str)
        .map(|value| value.strip_prefix("sha256:").unwrap_or(value).to_ascii_lowercase())
}

pub fn search_hugging_face(query: &str, limit: usize) -> Result<Vec<RemoteModel>, String> {
    if query.trim().is_empty() { return Err("search query must not be empty".into()); }
    if !(1..=50).contains(&limit) { return Err("--limit must be between 1 and 50".into()); }
    let url = format!("https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}&blobs=true",
        url_encode(query.trim()), limit);
    let body = get_https_text(&url)?;
    let value: Value = serde_json::from_str(&body).map_err(|e| format!("invalid Hugging Face model response: {e}"))?;
    let entries = value.as_array().ok_or("Hugging Face returned an unexpected model list")?;
    let mut results = Vec::new();
    for model in entries {
        let id = model.get("modelId").or_else(|| model.get("id")).and_then(Value::as_str).unwrap_or("(unknown repo)");
        let downloads = model.get("downloads").and_then(Value::as_u64).unwrap_or(0);
        // The list endpoint returns file names, but size and LFS digest metadata are
        // available only from the individual repository endpoint.
        let detail_url = format!("https://huggingface.co/api/models/{}?blobs=true&cardData=true", url_encode(id));
        let detail = get_https_text(&detail_url).ok().and_then(|body| serde_json::from_str::<Value>(&body).ok());
        let detail = detail.as_ref().unwrap_or(model);
        let license = detail.pointer("/cardData/license").or_else(|| detail.get("license"))
            .and_then(Value::as_str).unwrap_or("license not declared in model metadata").to_string();
        let gated = detail.get("gated").and_then(Value::as_bool).unwrap_or(false)
            || detail.get("gated").and_then(Value::as_str).map(|x| x != "false").unwrap_or(false);
        let files = detail.get("siblings").and_then(Value::as_array);
        let artifacts: Vec<_> = files.into_iter().flatten().filter_map(|file| {
            let name = file.get("rfilename")?.as_str()?;
            if !name.to_ascii_lowercase().ends_with(".gguf") { return None; }
            let size = file.get("size").and_then(Value::as_u64).or_else(|| file.pointer("/lfs/size").and_then(Value::as_u64));
            let sha256 = sha256_from_metadata(file);
            Some(RemoteArtifact { path: name.to_string(), size_bytes: size, sha256 })
        }).collect();
        results.push(RemoteModel { repository: id.to_string(), downloads, license, gated, artifacts });
    }
    Ok(results)
}

fn valid_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else { return false; };
    [owner, name].iter().all(|p| !p.is_empty() && *p != "." && *p != ".." && p.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)))
}

fn valid_repo_file(file: &str) -> bool {
    file.to_ascii_lowercase().ends_with(".gguf") && !file.starts_with('/') && !file.contains('\\') && !file.bytes().any(|b| b.is_ascii_control()) &&
        file.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
}

fn model_dir() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("KILN_MODELS_DIR") { return Ok(PathBuf::from(path)); }
    #[cfg(target_os = "windows")]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return Ok(PathBuf::from(path).join("KILN/models"));
    }
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
        .or(home_dir().map(|p| p.join(".local/share")))
        .ok_or("could not determine a user data directory; set KILN_MODELS_DIR")?;
    Ok(base.join("kiln/models"))
}

pub fn pull_hugging_face(repo: &str, file: &str) -> Result<PathBuf, String> {
    pull_hugging_face_with_progress(repo, file, &AtomicBool::new(false), |_, _| true)
}

/// Download and verify one public GGUF while reporting byte progress. Returning
/// false from the progress callback cancels the transfer and removes its partial file.
pub fn pull_hugging_face_with_progress(
    repo: &str,
    file: &str,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, Option<u64>) -> bool,
) -> Result<PathBuf, String> {
    if !valid_repo_id(repo) { return Err("model must be a Hugging Face repo in owner/name form".into()); }
    if !valid_repo_file(file) { return Err("--file must name a .gguf file inside the repository".into()); }

    let info_url = format!("https://huggingface.co/api/models/{}?blobs=true", url_encode(repo));
    let info: Value = serde_json::from_str(&get_https_text(&info_url)?)
        .map_err(|e| format!("invalid model metadata: {e}"))?;
    let gated = info.get("gated").is_some_and(|value| match value {
        Value::Bool(gated) => *gated,
        Value::String(gated) => gated != "false",
        _ => false,
    });
    if gated {
        return Err("this repository is gated and requires Hugging Face authorization; KILN will not bypass that access requirement".into());
    }
    let siblings = info.get("siblings").and_then(Value::as_array).ok_or("model metadata contains no file list")?;
    let sibling = siblings.iter().find(|item| item.get("rfilename").and_then(Value::as_str) == Some(file))
        .ok_or("the selected GGUF file was not found in that public repository")?;
    let expected = sha256_from_metadata(sibling)
        .ok_or("Hugging Face did not provide a SHA-256 digest for this file; refusing an unverifiable download")?;
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid SHA-256 digest".into());
    }
    let expected_size = sibling.get("size").and_then(Value::as_u64)
        .or_else(|| sibling.pointer("/lfs/size").and_then(Value::as_u64));
    const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024 * 1024;
    if expected_size.is_some_and(|size| size > MAX_MODEL_BYTES) {
        return Err("selected model exceeds the 2 TiB download limit".into());
    }

    let root = model_dir()?;
    let repo_dir = repo.replace('/', "--");
    let relative = Path::new(file);
    let destination = root.join(repo_dir).join(relative);
    if destination.exists() { return Err(format!("destination already exists: {}", destination.display())); }
    let parent = destination.parent().ok_or("invalid destination path")?;
    fs::create_dir_all(parent).map_err(|e| format!("could not create model directory: {e}"))?;
    let temp = destination.with_file_name(format!("{}.{}.part", relative.file_name().and_then(|x| x.to_str()).unwrap_or("model.gguf"), std::process::id()));
    if temp.exists() { return Err(format!("temporary download already exists: {}", temp.display())); }

    let encoded_file = file.split('/').map(url_encode).collect::<Vec<_>>().join("/");
    let url = format!("https://huggingface.co/{}/resolve/main/{}?download=true", url_encode(repo), encoded_file);
    let mut created_temp = false;
    let result = (|| {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(24 * 60 * 60))
            .redirect(https_redirect_policy())
            .build().map_err(|e| format!("could not initialize HTTPS client: {e}"))?;
        let mut response = client.get(&url).send()
            .map_err(|e| format!("model download request failed: {e}"))?
            .error_for_status().map_err(|e| format!("model download request failed: {e}"))?;
        if let Some(actual_size) = response.content_length() {
            if actual_size > MAX_MODEL_BYTES { return Err("selected model exceeds the 2 TiB download limit".into()); }
            if expected_size.is_some_and(|expected| expected != actual_size) {
                return Err(format!("model size changed: expected {} bytes, server reports {} bytes", expected_size.unwrap(), actual_size));
            }
        }
        let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&temp)
            .map_err(|e| format!("could not create temporary model file: {e}"))?;
        created_temp = true;
        let mut hasher = Sha256::new();
        let mut received = 0u64;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            if cancelled.load(Ordering::Relaxed) { return Err("model download cancelled".into()); }
            let count = response.read(&mut buffer).map_err(|e| format!("model download interrupted: {e}"))?;
            if count == 0 { break; }
            received = received.saturating_add(count as u64);
            if received > MAX_MODEL_BYTES || expected_size.is_some_and(|size| received > size) {
                return Err("download exceeded the advertised model size".into());
            }
            output.write_all(&buffer[..count]).map_err(|e| format!("could not write model data: {e}"))?;
            hasher.update(&buffer[..count]);
            if !progress(received, expected_size.or(response.content_length())) {
                cancelled.store(true, Ordering::Relaxed);
                return Err("model download cancelled".into());
            }
        }
        output.sync_all().map_err(|e| format!("could not flush downloaded model: {e}"))?;
        drop(output);
        if expected_size.is_some_and(|size| size != received) {
            return Err(format!("download size mismatch: expected {} bytes, received {} bytes", expected_size.unwrap(), received));
        }
        let actual = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        if actual != expected {
            return Err(format!("download SHA-256 mismatch: expected {expected}, got {actual}"));
        }
        if let Err(error) = GgufFile::open(&temp) {
            return Err(format!("downloaded file is not a readable KILN-supported GGUF: {error}"));
        }
        fs::hard_link(&temp, &destination).map_err(|e| format!("could not install downloaded model without overwriting an existing file: {e}"))?;
        let _ = fs::remove_file(&temp);
        Ok(destination.clone())
    })();
    if result.is_err() && created_temp { let _ = fs::remove_file(&temp); }
    result
}

/// Async counterpart for the UI/API server. Stream reads are cancellable while
/// the connection is idle, and progress callbacks never block a runtime worker.
pub async fn pull_hugging_face_async_with_progress(
    repo: &str,
    file: &str,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, Option<u64>) -> bool,
) -> Result<PathBuf, String> {
    if !valid_repo_id(repo) { return Err("model must be a Hugging Face repo in owner/name form".into()); }
    if !valid_repo_file(file) { return Err("--file must name a .gguf file inside the repository".into()); }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(24 * 60 * 60))
        .redirect(https_redirect_policy())
        .build().map_err(|e| format!("could not initialize HTTPS client: {e}"))?;
    let info_url = format!("https://huggingface.co/api/models/{}?blobs=true", url_encode(repo));
    let info_text = client.get(&info_url).send().await
        .map_err(|e| format!("Hugging Face request failed: {e}"))?
        .error_for_status().map_err(|e| format!("Hugging Face request failed: {e}"))?
        .text().await.map_err(|e| format!("Hugging Face returned invalid response text: {e}"))?;
    let info: Value = serde_json::from_str(&info_text).map_err(|e| format!("invalid model metadata: {e}"))?;
    let gated = info.get("gated").is_some_and(|value| match value {
        Value::Bool(gated) => *gated,
        Value::String(gated) => gated != "false",
        _ => false,
    });
    if gated {
        return Err("this repository is gated and requires Hugging Face authorization; KILN will not bypass that access requirement".into());
    }
    let siblings = info.get("siblings").and_then(Value::as_array).ok_or("model metadata contains no file list")?;
    let sibling = siblings.iter().find(|item| item.get("rfilename").and_then(Value::as_str) == Some(file))
        .ok_or("the selected GGUF file was not found in that public repository")?;
    let expected = sha256_from_metadata(sibling)
        .ok_or("Hugging Face did not provide a SHA-256 digest for this file; refusing an unverifiable download")?;
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid SHA-256 digest".into());
    }
    let expected_size = sibling.get("size").and_then(Value::as_u64)
        .or_else(|| sibling.pointer("/lfs/size").and_then(Value::as_u64));
    const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024 * 1024;
    if expected_size.is_some_and(|size| size > MAX_MODEL_BYTES) {
        return Err("selected model exceeds the 2 TiB download limit".into());
    }

    let root = model_dir()?;
    let destination = root.join(repo.replace('/', "--")).join(Path::new(file));
    if destination.exists() { return Err(format!("destination already exists: {}", destination.display())); }
    let parent = destination.parent().ok_or("invalid destination path")?;
    fs::create_dir_all(parent).map_err(|e| format!("could not create model directory: {e}"))?;
    let temp = destination.with_file_name(format!("{}.{}.part", Path::new(file).file_name().and_then(|x| x.to_str()).unwrap_or("model.gguf"), std::process::id()));
    if temp.exists() { return Err(format!("temporary download already exists: {}", temp.display())); }

    let encoded_file = file.split('/').map(url_encode).collect::<Vec<_>>().join("/");
    let url = format!("https://huggingface.co/{}/resolve/main/{}?download=true", url_encode(repo), encoded_file);
    let mut created_temp = false;
    let result = async {
        if cancelled.load(Ordering::Relaxed) { return Err("model download cancelled".into()); }
        let mut response = client.get(&url).send().await
            .map_err(|e| format!("model download request failed: {e}"))?
            .error_for_status().map_err(|e| format!("model download request failed: {e}"))?;
        if let Some(actual_size) = response.content_length() {
            if actual_size > MAX_MODEL_BYTES { return Err("selected model exceeds the 2 TiB download limit".into()); }
            if expected_size.is_some_and(|expected| expected != actual_size) {
                return Err(format!("model size changed: expected {} bytes, server reports {} bytes", expected_size.unwrap(), actual_size));
            }
        }
        let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&temp)
            .map_err(|e| format!("could not create temporary model file: {e}"))?;
        created_temp = true;
        let mut hasher = Sha256::new();
        let mut received = 0u64;
        loop {
            if cancelled.load(Ordering::Relaxed) { return Err("model download cancelled".into()); }
            let next_chunk = tokio::time::timeout(Duration::from_secs(30), response.chunk()).await
                .map_err(|_| "model download stalled for 30 seconds".to_string())?
                .map_err(|e| format!("model download interrupted: {e}"))?;
            let Some(chunk) = next_chunk else { break; };
            received = received.saturating_add(chunk.len() as u64);
            if received > MAX_MODEL_BYTES || expected_size.is_some_and(|size| received > size) {
                return Err("download exceeded the advertised model size".into());
            }
            output.write_all(&chunk).map_err(|e| format!("could not write model data: {e}"))?;
            hasher.update(&chunk);
            if !progress(received, expected_size.or(response.content_length())) {
                cancelled.store(true, Ordering::Relaxed);
                return Err("model download cancelled".into());
            }
        }
        output.sync_all().map_err(|e| format!("could not flush downloaded model: {e}"))?;
        drop(output);
        if expected_size.is_some_and(|size| size != received) {
            return Err(format!("download size mismatch: expected {} bytes, received {} bytes", expected_size.unwrap(), received));
        }
        let actual = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        if actual != expected { return Err(format!("download SHA-256 mismatch: expected {expected}, got {actual}")); }
        if let Err(error) = GgufFile::open(&temp) {
            return Err(format!("downloaded file is not a readable KILN-supported GGUF: {error}"));
        }
        fs::hard_link(&temp, &destination).map_err(|e| format!("could not install downloaded model without overwriting an existing file: {e}"))?;
        let _ = fs::remove_file(&temp);
        Ok(destination.clone())
    }.await;
    if result.is_err() && created_temp { let _ = fs::remove_file(&temp); }
    result
}

#[cfg(test)]
mod tests {
    use super::{sha256_from_metadata, valid_repo_file, valid_repo_id};
    use serde_json::json;

    #[cfg(unix)]
    #[test]
    fn snapshot_symlink_discovers_alias_without_a_duplicate_weight_file() {
        use super::collect_files;
        use std::collections::{HashMap, HashSet};
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "kiln-hf-zero-copy-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        let snapshot = root.join("models--vendor--generic/snapshots/abc123");
        let blobs = root.join("models--vendor--generic/blobs");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("weight");
        std::fs::write(&blob,b"GGUF fixture").unwrap();
        symlink("../../blobs/weight",snapshot.join("model.gguf")).unwrap();
        let mut visited=0;
        let mut seen=HashSet::new();
        let mut aliases=HashMap::new();
        let mut files=Vec::new();
        collect_files(&root,0,&mut visited,&mut seen,&mut aliases,&mut files);
        let canonical=std::fs::canonicalize(blob).unwrap();
        assert_eq!(files.len(),1);
        assert_eq!(files[0],canonical);
        assert_eq!(aliases.get(&canonical).unwrap(),&vec![
            "vendor/generic/model.gguf".to_string()
        ]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cache_snapshot_names_preserve_human_repo_and_file_identity_without_weight_copy() {
        use super::huggingface_snapshot_name;
        let path = std::path::Path::new(
            "/home/example/.cache/huggingface/hub/models--author--project/snapshots/abcdef1234/quant/model.Q4_K_M.gguf"
        );
        assert_eq!(huggingface_snapshot_name(path).as_deref(),
            Some("author/project/quant/model.Q4_K_M.gguf"));
        assert!(huggingface_snapshot_name(std::path::Path::new(
            "/tmp/models--author--project/blobs/sha256-other")).is_none());
        assert!(huggingface_snapshot_name(std::path::Path::new(
            "/tmp/models--author--project/snapshots/not-a-commit/weights.gguf")).is_none());
        assert!(huggingface_snapshot_name(std::path::Path::new(
            "/tmp/models--author--project/snapshots/012abc/file.bin")).is_none());
    }

    #[test]
    fn ollama_manifest_resolves_real_model_name_from_existing_blob_without_copying() {
        use super::ollama_names_for_roots;
        let root=std::env::temp_dir().join(format!(
            "kiln-ollama-alias-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        let manifests=root.join("manifests/registry.ollama.ai/library/fixture");
        let blobs=root.join("blobs");
        std::fs::create_dir_all(&manifests).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let digest="a".repeat(64);
        let blob=blobs.join(format!("sha256-{digest}"));
        std::fs::write(&blob,b"GGUF test").unwrap();
        let manifest=serde_json::json!({
            "schemaVersion":2,
            "layers":[
                {"mediaType":"application/vnd.ollama.image.model",
                 "digest":format!("sha256:{digest}"),"size":9}
            ]
        });
        std::fs::write(manifests.join("latest"),serde_json::to_vec(&manifest).unwrap()).unwrap();
        let mapping=ollama_names_for_roots(&[root.clone()]);
        assert_eq!(mapping.get(&blob).unwrap(),&vec!["fixture:latest".to_string()]);
        assert_eq!(std::fs::read(&blob).unwrap(),b"GGUF test");
        std::fs::write(manifests.join("bad"),br#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:../escape"}]}"#).unwrap();
        assert_eq!(ollama_names_for_roots(&[root.clone()]).len(),1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_library_rejects_traversal_and_non_gguf_files() {
        assert!(valid_repo_id("owner/model-name"));
        assert!(!valid_repo_id("../model"));
        assert!(!valid_repo_id("owner/../model"));
        assert!(valid_repo_file("nested/model.Q4_K_M.gguf"));
        assert!(!valid_repo_file("../outside.gguf"));
        assert!(!valid_repo_file("model.bin"));
    }

    #[test]
    fn reads_current_hugging_face_lfs_sha256_metadata() {
        let metadata = json!({"lfs":{"sha256":"8030F04528538D47BDA434F6F0BDF3952C40A58123E4D5E755332F23731A8684","size":105454144}});
        assert_eq!(
            sha256_from_metadata(&metadata).as_deref(),
            Some("8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684"),
        );
    }

    #[test]
    fn supports_legacy_hugging_face_digest_fields() {
        let oid = json!({"lfs":{"oid":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}});
        let top_level = json!({"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"});
        assert_eq!(sha256_from_metadata(&oid).as_deref(), Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        assert_eq!(sha256_from_metadata(&top_level).as_deref(), Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
    }
}

