//! Explicit opt-in one-binary, one-original-GGUF experimental MoE streaming.
//! The child inference process receives its own native kernel environment;
//! ordinary API sessions, other models and the parent environment stay intact.
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use kiln_models::{
    discover_complete_fused_moe_selections, resolve_fused_expert_tensor_components,
    ExpertTensorComponent, GgufFile,
};

const MAX_STREAM_INDEX_BYTES: usize = 64 * 1024 * 1024;

fn external_index_contents(components: &[ExpertTensorComponent]) -> Result<Vec<u8>, String> {
    if components.is_empty() {
        return Err("cannot stream an empty expert layout".into());
    }
    let mut lines = String::from("KILN-MOE-STREAM-1\n");
    let mut seen = HashSet::new();
    for entry in components {
        let suffix = format!("#expert:{}", entry.expert_id);
        let name = entry.tensor_name.strip_suffix(&suffix)
            .ok_or_else(|| format!("invalid fused component {}", entry.tensor_name))?;
        if name.is_empty() || name.len() >= 64 ||
            !name.bytes().all(|c| c.is_ascii_alphanumeric() ||
                matches!(c, b'.' | b'_' | b'-')) {
            return Err("fused tensor identity cannot be represented safely".into());
        }
        if entry.byte_length == 0 ||
            entry.source_byte_offset.checked_add(entry.byte_length).is_none() {
            return Err("invalid expert slab byte range".into());
        }
        if !seen.insert((name.to_owned(), entry.expert_id)) {
            return Err("ambiguous duplicate expert source index".into());
        }
        let line = format!("{name}\t{}\t{}\t{}\n",
            entry.expert_id, entry.source_byte_offset, entry.byte_length);
        if lines.len().checked_add(line.len())
            .is_none_or(|count| count > MAX_STREAM_INDEX_BYTES) {
            return Err("stream index exceeds 64 MiB".into());
        }
        lines.push_str(&line);
    }
    Ok(lines.into_bytes())
}

struct TemporaryStreamIndex {
    path: PathBuf,
}

impl TemporaryStreamIndex {
    fn create(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_STREAM_INDEX_BYTES {
            return Err("invalid external expert stream index length".into());
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        for attempt in 0..8 {
            let path = std::env::temp_dir().join(format!(
                "kiln-moe-index-{}-{nanos}-{attempt}.tsv", std::process::id()
            ));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut handle) => {
                    if let Err(error) = handle.write_all(bytes).and_then(|_| handle.sync_all()) {
                        drop(handle);
                        let _ = std::fs::remove_file(path);
                        return Err(format!("stream index write: {error}"));
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("stream index file creation: {error}")),
            }
        }
        Err("unable to allocate a unique temporary stream index".into())
    }
}

impl Drop for TemporaryStreamIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// User-visible experimental launcher: the same single KILN executable is
/// re-executed with a session-scoped source-GGUF expert index. No external
/// program, model copy, permanent manifest or daemon is required.
pub fn run(
    model: &str,
    max_tokens: usize,
    keep_context: bool,
    lazy_moe_experts: bool,
    cache_mib: u32,
    parallel_reads: bool,
) -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err("native on-demand CPU expert streaming currently requires Linux".into());
    }
    if max_tokens == 0 || cache_mib > 512 {
        return Err("max-tokens must be positive and cache-mib must be 0..=512".into());
    }
    let found = if model == "auto" {
        crate::model_library::resolve_auto_model()?
    } else {
        crate::model_library::resolve_model_identifier(model)?
    };
    let path = std::fs::canonicalize(&found)
        .map_err(|error| format!("model path cannot be resolved: {error}"))?;
    let model_data = GgufFile::open(&path).map_err(|e| e.to_string())?;
    let choices = discover_complete_fused_moe_selections(&model_data)
        .map_err(|e| format!(
            "model cannot use automatic native MoE streaming ({e}); use explicit expert-export-gguf-stream-index for unfamiliar layouts"
        ))?;
    let source_map = resolve_fused_expert_tensor_components(
        &model_data.tensors, model_data.data_offset, model_data.file_size(), &choices
    ).map_err(|e| e.to_string())?;
    let bytes = external_index_contents(&source_map.components)?;
    drop(model_data);
    let temp = TemporaryStreamIndex::create(&bytes)?;
    let binary = std::env::current_exe()
        .map_err(|e| format!("unable to locate current KILN binary: {e}"))?;

    eprintln!(
        "KILN experimental CPU MoE: {} verified tensors; source GGUF unchanged; cache {} MiB; parallel reads {}; deployment acceptance NOT established",
        choices.len(), cache_mib, parallel_reads,
    );
    let mut command = Command::new(binary);
    command.arg("run").arg(&path)
        .arg("--native-greedy")
        .arg("--max-tokens").arg(max_tokens.to_string())
        .env("KILN_MOE_STREAM_BLOB", &path)
        .env("KILN_MOE_STREAM_INDEX", &temp.path)
        .env("KILN_MOE_STREAM_CACHE_MIB", cache_mib.to_string())
        .env("KILN_MOE_STREAM_PARALLEL", if parallel_reads { "1" } else { "0" });
    if keep_context {
        command.arg("--keep-context");
    }
    if lazy_moe_experts {
        command.arg("--lazy-moe-experts");
    }
    let exit = command.status()
        .map_err(|e| format!("failed to execute KILN native inference child: {e}"))?;
    if !exit.success() {
        return Err(format!(
            "native MoE streaming child exited with {}",
            exit.code().map_or_else(|| "signal".into(), |code| code.to_string())
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn safe_external_index_is_bounded_and_rejects_duplicate_or_ambiguous_experts() {
        let component = ExpertTensorComponent {
            layer: 2, expert_id: 1,
            tensor_name: "blk.2.ffn_up_exps.weight#expert:1".into(),
            source_byte_offset: 8192, byte_length: 512,
        };
        assert_eq!(external_index_contents(&[component.clone()]).unwrap(),
            b"KILN-MOE-STREAM-1\nblk.2.ffn_up_exps.weight\t1\t8192\t512\n");
        assert!(external_index_contents(&[component.clone(), component.clone()]).is_err());
        assert!(external_index_contents(&[ExpertTensorComponent {
            tensor_name: "../bad#expert:1".into(), ..component.clone()
        }]).is_err());
        assert!(external_index_contents(&[ExpertTensorComponent {
            source_byte_offset: u64::MAX, byte_length: 10, ..component.clone()
        }]).is_err());
        assert!(external_index_contents(&[]).is_err());
    }

    #[test]
    fn temporary_session_index_is_removed_on_drop() {
        let filename = {
            let file = TemporaryStreamIndex::create(b"KILN-MOE-STREAM-1\n").unwrap();
            assert_eq!(std::fs::read(&file.path).unwrap(), b"KILN-MOE-STREAM-1\n");
            file.path.clone()
        };
        assert!(!Path::new(&filename).exists());
    }
}
