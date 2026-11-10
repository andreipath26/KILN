//! Human-readable CLI adapters for the shared model-library service.

use std::path::PathBuf;

use kiln_models::library;

pub fn resolve_auto_model() -> Result<PathBuf, String> {
    let models = library::discover_models(&[]);
    match models.as_slice() {
        [model] => library::normalize_gguf_load_path(std::path::Path::new(&model.path)),
        [] => Err("no readable local GGUF was found; use `kiln search <query>` and `kiln pull <owner/repo> --file <file.gguf>`, or pass a model path".into()),
        many => {
            let choices = many.iter().map(|model| format!("  {} ({:.2} GiB)\n    {}", model.name,
                model.size_bytes as f64 / 1_073_741_824.0, model.path)).collect::<Vec<_>>().join("\n");
            Err(format!("found {} local GGUF candidates; choose one with `kiln run <path>`:\n{}", many.len(), choices))
        }
    }
}

/// Resolve a readable model file OR an unambiguous discovered GGUF/Ollama
/// model label. Never execute Ollama or duplicate its content-addressed blobs.
pub fn resolve_model_identifier(input: &str) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(input);
    if candidate.is_file() {
        return library::normalize_gguf_load_path(&candidate);
    }
    let models = library::discover_models(&[]);
    resolve_named_model(&models, input)
}

fn resolve_named_model(models: &[library::LocalModel], input: &str) -> Result<PathBuf, String> {
    let matching = models.iter().filter(|model| {
        model.name.eq_ignore_ascii_case(input)
            || model.aliases.iter().any(|name| name.eq_ignore_ascii_case(input))
    }).collect::<Vec<_>>();
    match matching.as_slice() {
        [model] => Ok(PathBuf::from(&model.path)),
        [] => Err(format!(
            "model {input:?} was not found as a readable GGUF path or discovered model name; use `kiln list` to see installed models"
        )),
        many => Err(format!(
            "model label {input:?} is ambiguous ({} files); pass an exact GGUF path",
            many.len()
        )),
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
            let alternative_tags: Vec<_> = model.aliases.iter()
                .filter(|tag| *tag != &model.name).collect();
            if !alternative_tags.is_empty() {
                println!("  alternate installed names: {}", alternative_tags.into_iter()
                    .map(String::as_str).collect::<Vec<_>>().join(", "));
            }
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


#[cfg(test)]
mod model_identifier_tests {
    use super::*;
    use kiln_models::library::LocalModel;
    fn fixture(name: &str, path: &str) -> LocalModel {
        LocalModel {
            name: name.into(), path: path.into(), size_bytes: 4096,
            source: "Ollama manifest".into(), valid_gguf: true,
            aliases: vec![name.to_owned()],
        }
    }
    #[test]
    fn rejects_ambiguous_names_instead_of_choosing_an_arbitrary_blob() {
        let mut a = fixture("model:latest","/tmp/model-a");
        a.aliases.push("alternative:v1".into());
        let models = [a, fixture("MODEL:latest","/tmp/model-b")];
        assert!(resolve_named_model(&models, "model:latest").is_err());
        assert_eq!(resolve_named_model(&models, "alternative:v1").unwrap(),
            PathBuf::from("/tmp/model-a"));
        assert!(resolve_named_model(&models, "other:latest").is_err());
    }
}
