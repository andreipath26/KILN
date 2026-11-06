//! Local GGUF discovery and Hugging Face GGUF search/download helpers.

use kiln_models::gguf::GgufFile;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_SCAN_DEPTH: usize = 12;
const MAX_SCAN_ENTRIES: usize = 100_000;

#[derive(Debug, Serialize)]
struct LocalModel {
    name: String,
    path: String,
    size_bytes: u64,
    source: String,
    valid_gguf: bool,
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

fn collect_files(root: &Path, depth: usize, visited: &mut usize, seen: &mut HashSet<PathBuf>, out: &mut Vec<PathBuf>) {
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
                collect_files(&path, depth + 1, visited, seen, out);
            }
        } else if meta.is_file() {
            let looks_like_gguf = path.extension().and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("gguf")).unwrap_or(false);
            // Ollama stores GGUF weights as extensionless content-addressed blobs.
            if looks_like_gguf || has_gguf_magic(&path) {
                let canonical = fs::canonicalize(&path).unwrap_or(path);
                if seen.insert(canonical.clone()) { out.push(canonical); }
            }
        }
    }
}

fn discover_models(extra_roots: &[PathBuf]) -> Vec<LocalModel> {
    let mut roots = standard_roots();
    roots.extend(extra_roots.iter().cloned());
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    let mut unique_roots = HashSet::new();
    let mut visited = 0usize;
    for root in roots {
        if root.is_dir() {
            let canonical = fs::canonicalize(&root).unwrap_or(root);
            if unique_roots.insert(canonical.clone()) {
                collect_files(&canonical, 0, &mut visited, &mut seen, &mut files);
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
        let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("model.gguf").to_owned();
        models.push(LocalModel {
            name,
            path: path.display().to_string(),
            size_bytes: metadata.len(),
            source: root_label(&path),
            valid_gguf: true,
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

pub fn list_models(extra_roots: &[PathBuf], json: bool) -> Result<(), String> {
    let models = discover_models(extra_roots);

    if json {
        println!("{}", serde_json::to_string_pretty(&models).map_err(|e| e.to_string())?);
    } else if models.is_empty() {
        println!("No readable GGUF models found.");
        println!("Set KILN_MODELS_DIR or pass --path to scan another directory.");
    } else {
        for model in models {
            println!("{}  ({:.2} GiB)\n  {}\n  discovered in {}", model.name,
                model.size_bytes as f64 / 1_073_741_824.0, model.path, model.source);
        }
    }
    Ok(())
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
    let output = Command::new("curl")
        .args(["--fail", "--location", "--proto-redir", "=https", "--silent", "--show-error", "--max-time", "60", url])
        .output().map_err(|e| format!("could not start curl (required for model search/download): {e}"))?;
    if !output.status.success() {
        return Err(format!("Hugging Face request failed: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("Hugging Face returned non-UTF-8 JSON: {e}"))
}

pub fn search_hugging_face(query: &str, limit: usize) -> Result<(), String> {
    if query.trim().is_empty() { return Err("search query must not be empty".into()); }
    if !(1..=50).contains(&limit) { return Err("--limit must be between 1 and 50".into()); }
    let url = format!("https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}&blobs=true",
        url_encode(query.trim()), limit);
    let body = get_https_text(&url)?;
    let value: Value = serde_json::from_str(&body).map_err(|e| format!("invalid Hugging Face model response: {e}"))?;
    let entries = value.as_array().ok_or("Hugging Face returned an unexpected model list")?;
    let mut count = 0usize;
    for model in entries {
        let id = model.get("modelId").or_else(|| model.get("id")).and_then(Value::as_str).unwrap_or("(unknown repo)");
        let downloads = model.get("downloads").and_then(Value::as_u64).unwrap_or(0);
        // The list endpoint returns file names, but size and LFS digest metadata are
        // available only from the individual repository endpoint.
        let detail_url = format!("https://huggingface.co/api/models/{}?blobs=true&cardData=true", url_encode(id));
        let detail = get_https_text(&detail_url).ok().and_then(|body| serde_json::from_str::<Value>(&body).ok());
        let detail = detail.as_ref().unwrap_or(model);
        let license = detail.pointer("/cardData/license").or_else(|| detail.get("license"))
            .and_then(Value::as_str).unwrap_or("license not declared in model metadata");
        let files = detail.get("siblings").and_then(Value::as_array);
        let ggufs: Vec<_> = files.into_iter().flatten().filter_map(|file| {
            let name = file.get("rfilename")?.as_str()?;
            if !name.to_ascii_lowercase().ends_with(".gguf") { return None; }
            let size = file.get("size").and_then(Value::as_u64).or_else(|| file.pointer("/lfs/size").and_then(Value::as_u64));
            Some((name, size))
        }).collect();
        if ggufs.is_empty() {
            println!("{:?}  downloads={downloads}  (GGUF files not listed by search API)", id);
        } else {
            for (file, size) in ggufs {
                let size_text = size.map(|n| format!("{:.2} GiB", n as f64 / 1_073_741_824.0)).unwrap_or_else(|| "size unknown".into());
                println!("{:?} --file {:?}  {size_text}  license={:?}  downloads={downloads}", id, file, license);
            }
        }
        count += 1;
    }
    if count == 0 { println!("No GGUF repositories matched. Try another search term."); }
    Ok(())
}

fn valid_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else { return false; };
    [owner, name].iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)))
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

fn sha256(path: &Path) -> Result<String, String> {
    #[cfg(target_os = "windows")]
    let output = Command::new("certutil").args(["-hashfile", path.to_string_lossy().as_ref(), "SHA256"]).output();
    #[cfg(target_os = "macos")]
    let output = Command::new("shasum").args(["-a", "256"]).arg(path).output();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let output = Command::new("sha256sum").arg(path).output();
    let output = output.map_err(|e| format!("could not run a SHA-256 utility: {e}"))?;
    if !output.status.success() { return Err("SHA-256 utility failed".into()); }
    let text = String::from_utf8_lossy(&output.stdout);
    let hash = text.lines().find_map(|line| {
        let candidate = line.trim().split_whitespace().next()?;
        (candidate.len() == 64 && candidate.bytes().all(|b| b.is_ascii_hexdigit())).then(|| candidate.to_ascii_lowercase())
    }).ok_or("could not parse SHA-256 utility output")?;
    Ok(hash)
}

pub fn pull_hugging_face(repo: &str, file: &str) -> Result<PathBuf, String> {
    if !valid_repo_id(repo) { return Err("model must be a Hugging Face repo in owner/name form".into()); }
    if !valid_repo_file(file) { return Err("--file must name a .gguf file inside the repository".into()); }

    let info_url = format!("https://huggingface.co/api/models/{}?blobs=true", url_encode(repo));
    let info: Value = serde_json::from_str(&get_https_text(&info_url)?)
        .map_err(|e| format!("invalid model metadata: {e}"))?;
    let siblings = info.get("siblings").and_then(Value::as_array).ok_or("model metadata contains no file list")?;
    let sibling = siblings.iter().find(|item| item.get("rfilename").and_then(Value::as_str) == Some(file))
        .ok_or("the selected GGUF file was not found in that public repository")?;
    let expected = sibling.pointer("/lfs/oid").or_else(|| sibling.get("sha256"))
        .and_then(Value::as_str).map(|x| x.strip_prefix("sha256:").unwrap_or(x).to_ascii_lowercase())
        .ok_or("Hugging Face did not provide a SHA-256 digest for this file; refusing an unverifiable download")?;
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid SHA-256 digest".into());
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
    let status = Command::new("curl").args(["--fail", "--location", "--proto-redir", "=https", "--show-error", "--retry", "3", "--retry-all-errors", "--progress-bar", "--output"])
        .arg(&temp).arg("--").arg(&url).status()
        .map_err(|e| format!("could not start curl (required for model download): {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&temp);
        return Err(format!("download failed with {status}"));
    }
    let actual = match sha256(&temp) {
        Ok(hash) => hash,
        Err(error) => { let _ = fs::remove_file(&temp); return Err(error); }
    };
    if actual != expected {
        let _ = fs::remove_file(&temp);
        return Err(format!("download SHA-256 mismatch: expected {expected}, got {actual}"));
    }
    if let Err(error) = GgufFile::open(&temp) {
        let _ = fs::remove_file(&temp);
        return Err(format!("downloaded file is not a readable KILN-supported GGUF: {error}"));
    }
    fs::rename(&temp, &destination).map_err(|e| {
        let _ = fs::remove_file(&temp);
        format!("could not install downloaded model: {e}")
    })?;
    Ok(destination)
}

