//! Human-readable CLI adapters for the shared model-library service.

use std::path::PathBuf;

use kiln_models::library;

pub fn resolve_auto_model() -> Result<PathBuf, String> {
    let models = library::discover_models(&[]);
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
    let models = library::discover_models(extra_roots);
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

pub fn search_hugging_face(query: &str, limit: usize) -> Result<(), String> {
    let results = library::search_hugging_face(query, limit)?;
    for model in &results {
        if model.artifacts.is_empty() {
            println!("{:?}  downloads={}  (GGUF files not listed by search API)", model.repository, model.downloads);
        }
        for artifact in &model.artifacts {
            let size = artifact.size_bytes.map(|n| format!("{:.2} GiB", n as f64 / 1_073_741_824.0))
                .unwrap_or_else(|| "size unknown".into());
            println!("{:?} --file {:?}  {}  license={:?}  downloads={}", model.repository, artifact.path, size, model.license, model.downloads);
        }
    }
    if results.is_empty() { println!("No GGUF repositories matched. Try another search term."); }
    Ok(())
}

pub fn pull_hugging_face(repo: &str, file: &str) -> Result<PathBuf, String> {
    library::pull_hugging_face(repo, file)
}

