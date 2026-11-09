//! Explicit opt-in one-binary, one-original-GGUF experimental MoE streaming.
//! The child inference process receives its own native kernel environment;
//! ordinary API sessions, other models and the parent environment stay intact.
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
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
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
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

/// Demand that every declared fused projection executed with selected
/// external expert operands. A successfully exited child is not sufficient:
/// unsupported layouts and failed pread must not be silently mistaken for
/// streamed inference. Missing, truncated or malformed audit => fail closed.
fn verify_stream_audit(contents: &str, expected_names: &HashSet<String>) -> Result<(u64, u64, u64), String> {
    if contents.len() > 16 * 1024 * 1024 {
        return Err("native stream audit exceeded the 16 MiB safety cap".into());
    }
    if contents.is_empty() || expected_names.is_empty() {
        return Err("native graph did not report selected external expert computation".into());
    }
    let mut observed = HashSet::new();
    let (mut bytes, mut pread_bytes, mut hits) = (0u64, 0u64, 0u64);
    for line in contents.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 6 {
            return Err("malformed native expert streaming audit".into());
        }
        let name = fields[0];
        if !expected_names.contains(name) {
            return Err(format!("unexpected native expert tensor {name:?}"));
        }
        let parse = |index: usize| -> Result<u64, String> {
            fields[index].parse::<u64>().map_err(|_| "native audit counter is invalid".into())
        };
        let requested = parse(1)?;
        let loaded = parse(2)?;
        let delivered = parse(3)?;
        let pread = parse(4)?;
        let cached = parse(5)?;
        if requested == 0 || loaded != requested || delivered == 0 || pread > delivered ||
            cached > loaded || pread == 0 && cached == 0 {
            return Err(format!("native expert {name} was not fully served externally"));
        }
        observed.insert(name.to_owned());
        bytes = bytes.checked_add(delivered).ok_or("expert delivered-byte counter overflow")?;
        pread_bytes = pread_bytes.checked_add(pread).ok_or("expert pread-byte counter overflow")?;
        hits = hits.checked_add(cached).ok_or("expert hot-cache hit counter overflow")?;
    }
    if observed != *expected_names {
        return Err(format!("external CPU expert graph executed {}/{} verified projections",
            observed.len(), expected_names.len()));
    }
    Ok((bytes, pread_bytes, hits))
}

/// Reject swapped/replaced model files between offset validation and
/// external read completion, including a same-length replacement. This
/// cannot replace actual cryptographic weight verification on every read.
#[cfg(unix)]
fn same_file_identity(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.dev() == after.dev() && before.ino() == after.ino()
}
#[cfg(not(unix))]
fn same_file_identity(_before: &std::fs::Metadata, _after: &std::fs::Metadata) -> bool {
    true
}

struct TemporaryAudit {
    path: PathBuf,
}
impl TemporaryAudit {
    fn create() -> Result<Self, String> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?.as_nanos();
        for attempt in 0..8 {
            let path = std::env::temp_dir().join(format!(
                "kiln-moe-audit-{}-{seed}-{attempt}.tsv", std::process::id()
            ));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => { drop(file); return Ok(Self { path }); }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("native audit setup failed: {error}")),
            }
        }
        Err("unable to allocate a unique temporary native audit".into())
    }
}
impl Drop for TemporaryAudit {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
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
    let source_before = std::fs::metadata(&path)
        .map_err(|e| format!("cannot stat source GGUF: {e}"))?;
    if !source_before.is_file() {
        return Err("source GGUF must be a regular file".into());
    }
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
    let audit = TemporaryAudit::create()?;
    let expected_names: HashSet<String> = choices.iter()
        .map(|selection| selection.tensor_name.clone()).collect();
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
        .env("KILN_MOE_STREAM_AUDIT", &audit.path)
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
    let audit_bytes = std::fs::metadata(&audit.path)
        .map_err(|e| format!("native MoE streaming audit unreadable: {e}"))?.len();
    if audit_bytes > 16 * 1024 * 1024 {
        return Err("native stream audit exceeded the 16 MiB safety cap".into());
    }
    let source_after = std::fs::metadata(&path)
        .map_err(|e| format!("source GGUF disappeared during streaming: {e}"))?;
    if source_before.len() != source_after.len()
        || source_before.modified().ok() != source_after.modified().ok()
        || !same_file_identity(&source_before, &source_after) {
        return Err("source GGUF file changed during streamed inference; discard results".into());
    }
    let recorded = std::fs::read_to_string(&audit.path)
        .map_err(|e| format!("could not verify native MoE graph read audit: {e}"))?;
    let (served, pread, hits) = verify_stream_audit(&recorded, &expected_names)?;
    eprintln!(
        "KILN external expert graph verified: {} tensors, {served} operand bytes, {pread} pread bytes (may be OS-cached), {hits} hot expert cache hits",
        expected_names.len()
    );
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
    fn native_cpu_audit_requires_all_projections_and_no_fallbacks() {
        let expected = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_exps.weight"].into_iter().map(str::to_owned).collect();
        let rows = "blk.0.ffn_up_exps.weight\t2\t2\t1024\t1024\t0\nblk.0.ffn_down_exps.weight\t2\t2\t1024\t0\t2\nblk.0.ffn_gate_exps.weight\t2\t2\t1024\t0\t2\n";
        assert_eq!(verify_stream_audit(rows, &expected).unwrap(), (3072,1024,4));
        assert!(verify_stream_audit("", &expected).is_err());
        assert!(verify_stream_audit(&rows.lines().take(2).collect::<Vec<_>>().join("\n"), &expected).is_err());
        assert!(verify_stream_audit(&rows.replace("\t2\t2\t1024\t1024\t0", "\t2\t1\t1024\t1024\t0"), &expected).is_err());
        assert!(verify_stream_audit(&rows.replace("\t0\t2", "\t0\t0"), &expected).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn temporary_expert_index_and_audit_are_owner_readable_only() {
        use std::os::unix::fs::PermissionsExt;
        let index = TemporaryStreamIndex::create(b"KILN-MOE-STREAM-1\n").unwrap();
        let audit = TemporaryAudit::create().unwrap();
        assert_eq!(std::fs::metadata(&index.path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&audit.path).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn temporary_session_index_is_removed_on_drop() {
        let filename = {
            let file = TemporaryStreamIndex::create(b"KILN-MOE-STREAM-1\n").unwrap();
            assert_eq!(std::fs::read(&file.path).unwrap(), b"KILN-MOE-STREAM-1\n");
            file.path.clone()
        };
        assert!(!std::path::Path::new(&filename).exists());
    }
}
