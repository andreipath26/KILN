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
const MAX_MULTI_SOURCE_SHARDS: usize = 64;
const MAX_MULTI_SOURCE_ROWS: usize = 300_000;

fn split_external_index_contents(
    index: &kiln_models::split_gguf::SplitGgufInventory,
) -> Result<(Vec<u8>, HashSet<String>), String> {
    if index.shard_count < 2 || index.shards.len() != index.shard_count as usize ||
        index.shards.len() > MAX_MULTI_SOURCE_SHARDS {
        return Err("native split streaming requires 2..=64 complete source shards".into());
    }
    if index.fused_expert_slabs.is_empty() ||
        !index.unsupported_expert_tensors.is_empty() ||
        index.fused_expert_slabs.len() > MAX_MULTI_SOURCE_ROWS {
        return Err("native split streaming needs a complete supported fused-expert layout (maximum 300000 slabs)".into());
    }
    let mut lines = format!("KILN-MOE-STREAM-2\nSOURCES\t{}\n",index.shards.len());
    for (id,shard) in index.shards.iter().enumerate() {
        if shard.shard_number != (id+1) as u32 {
            return Err("split GGUF shards are not consecutively numbered".into());
        }
        let name = shard.path.to_str()
            .ok_or("split source path is not valid UTF-8")?;
        if !name.starts_with('/') || name.len() >= 4096 ||
            name.bytes().any(|b| matches!(b,b'\t' | b'\r' | b'\n' | 0)) {
            return Err("split source path is not safe for the native index".into());
        }
        let line = format!("SOURCE\t{id}\t{name}\n");
        if lines.len().checked_add(line.len())
            .is_none_or(|n| n > MAX_STREAM_INDEX_BYTES) {
            return Err("split source index exceeds 64 MiB".into());
        }
        lines.push_str(&line);
    }
    lines.push_str("DATA\n");
    let mut seen = HashSet::new();
    let mut expected_names = HashSet::new();
    for slab in &index.fused_expert_slabs {
        let name = slab.tensor_name.as_str();
        if name.is_empty() || name.len() >= 64 ||
            !name.bytes().all(|c| c.is_ascii_alphanumeric() ||
                matches!(c,b'.'|b'_'|b'-')) {
            return Err("native tensor name is not representable in the verified index".into());
        }
        if !seen.insert((name.to_owned(),slab.expert_id)) {
            return Err("duplicate split source expert identity".into());
        }
        let shard_id = slab.shard_number.checked_sub(1)
            .ok_or("expert references shard zero")? as usize;
        let source = index.shards.get(shard_id)
            .ok_or("expert references an invalid source shard")?;
        if slab.source_path != source.path || slab.byte_length == 0 ||
            slab.source_byte_offset.checked_add(slab.byte_length)
                .is_none_or(|end| end > source.file_size_bytes) {
            return Err("split expert byte range does not match its validated source".into());
        }
        expected_names.insert(name.to_owned());
        let row = format!("{name}\t{}\t{shard_id}\t{}\t{}\n",
            slab.expert_id,slab.source_byte_offset,slab.byte_length);
        if lines.len().checked_add(row.len())
            .is_none_or(|n| n > MAX_STREAM_INDEX_BYTES) {
            return Err("split expert index exceeds 64 MiB".into());
        }
        lines.push_str(&row);
    }
    if expected_names.is_empty() { return Err("split index contains no fused MoE projection".into()); }
    Ok((lines.into_bytes(),expected_names))
}



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
        // A single native MUL_MAT_ID tensor has constant slab size for all
        // selected expert IDs. Require every claimed delivered operand to be
        // explained by either an external pread or exactly one hot-cache hit.
        // This makes malformed/inflated audit counters fail closed rather
        // than counting them as proof of streamed CPU computation.
        if delivered % loaded != 0 {
            return Err(format!("native expert {name} reported fractional slab bytes"));
        }
        let slab_bytes = delivered / loaded;
        let cached_bytes = slab_bytes.checked_mul(cached)
            .ok_or("native hot-cache bytes overflow")?;
        if pread.checked_add(cached_bytes) != Some(delivered) {
            return Err(format!("native expert {name} cache/read accounting does not match served bytes"));
        }
        // Native GGML appends one audit record per MUL_MAT_ID execution,
        // not one record per tensor. Repeated projection names are therefore
        // legitimate across tokens and batches. Validate every occurrence,
        // then aggregate counters while tracking distinct coverage.
        observed.insert(name.to_owned());
        bytes = bytes.checked_add(delivered).ok_or("expert delivered-byte counter overflow")?;
        pread_bytes = pread_bytes.checked_add(pread).ok_or("expert pread-byte counter overflow")?;
        hits = hits.checked_add(cached).ok_or("expert hot-cache hit counter overflow")?;
    }
    if observed != *expected_names {
        return Err(format!("external CPU expert graph executed {}/{} verified projections",
            observed.len(), expected_names.len()));
    }
    // The launcher starts a new child with an empty process-local hot cache.
    // An audit consisting entirely of cache hits is not credible evidence
    // that any selected expert was fetched from the source GGUF.
    if pread_bytes == 0 {
        return Err("native streaming audit contained no external expert reads".into());
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
    stdio_json: bool,
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
    if let Some((_, _, parts)) = kiln_models::library::split_gguf_parts(&path) {
        // The native expert read callback currently accepts one file
        // descriptor. Building a one-source index from shard 1 would omit
        // routed experts in other shards and falsely claim graph coverage.
        return Err(format!(
            "this model has {parts} GGUF shards; native multi-source expert reads are not yet wired. \
             Use 'kiln plan <first-shard.gguf>' to inspect all shard sources. \
             Do not select a partial single-shard stream index."
        ));
    }
    let source_before = std::fs::metadata(&path)
        .map_err(|e| format!("cannot stat source GGUF: {e}"))?;
    if !source_before.is_file() {
        return Err("source GGUF must be a regular file".into());
    }
    let split_inventory = kiln_models::split_gguf::inspect_split_gguf(&path)?;
    let (bytes,expected_names,source_snapshots) = if let Some(ref index) = split_inventory {
        let snapshots = index.shards.iter().map(|part| {
            std::fs::metadata(&part.path)
                .map(|meta|(part.path.clone(),meta))
                .map_err(|e|format!("cannot stat split source {}: {e}",part.path.display()))
        }).collect::<Result<Vec<_>,_>>()?;
        let (index_bytes,names) = split_external_index_contents(index)?;
        (index_bytes,names,snapshots)
    } else {
        let model_data = GgufFile::open(&path).map_err(|e| e.to_string())?;
        let choices = discover_complete_fused_moe_selections(&model_data)
            .map_err(|e| format!(
                "model cannot use automatic native MoE streaming ({e}); use explicit expert-export-gguf-stream-index for unfamiliar layouts"
            ))?;
        let source_map = resolve_fused_expert_tensor_components(
            &model_data.tensors,model_data.data_offset,model_data.file_size(),&choices
        ).map_err(|e|e.to_string())?;
        let bytes = external_index_contents(&source_map.components)?;
        let names = choices.iter().map(|choice|choice.tensor_name.clone()).collect();
        (bytes,names,vec![(path.clone(),source_before.clone())])
    };
    let temp = TemporaryStreamIndex::create(&bytes)?;
    let audit = TemporaryAudit::create()?;
    let binary = std::env::current_exe()
        .map_err(|e| format!("unable to locate current KILN binary: {e}"))?;

    eprintln!(
        "KILN experimental CPU MoE: {} verified tensors; source GGUF unchanged; cache {} MiB; parallel reads {}; deployment acceptance NOT established",
        expected_names.len(), cache_mib, parallel_reads,
    );
    let mut command = Command::new(binary);
    if stdio_json {
        command.arg("worker").arg(&path);
    } else {
        command.arg("run").arg(&path)
            .arg("--native-greedy")
            .arg("--max-tokens").arg(max_tokens.to_string());
    }
    command.env("KILN_MOE_STREAM_BLOB", &path)
        .env("KILN_MOE_STREAM_INDEX", &temp.path)
        .env("KILN_MOE_STREAM_AUDIT", &audit.path)
        .env("KILN_MOE_STREAM_CACHE_MIB", cache_mib.to_string())
        .env("KILN_MOE_STREAM_PARALLEL", if parallel_reads { "1" } else { "0" });
    if keep_context {
        command.arg("--keep-context");
    }
    if lazy_moe_experts || split_inventory.is_some() {
        // Split GGUFs larger than RAM must not eagerly populate all expert
        // pages while the selected-expert external reader is active.
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
    for (source, before) in &source_snapshots {
        let after = std::fs::metadata(source)
            .map_err(|e|format!("source GGUF shard {} disappeared: {e}",source.display()))?;
        if before.len() != after.len() ||
            before.modified().ok() != after.modified().ok() ||
            !same_file_identity(before,&after) {
            return Err(format!(
                "source GGUF shard {} changed during streamed inference; discard results",
                source.display()));
        }
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
        // GGML emits repeated records for a tensor across routed batches:
        // counters must include each execution, while coverage stays unique.
        assert_eq!(
            verify_stream_audit(&format!("{rows}{}\n", rows.lines().next().unwrap()), &expected).unwrap(),
            (4096, 2048, 4)
        );
        assert!(verify_stream_audit(&rows.lines().take(2).collect::<Vec<_>>().join("\n"), &expected).is_err());
        assert!(verify_stream_audit(&rows.replace("\t2\t2\t1024\t1024\t0", "\t2\t1\t1024\t1024\t0"), &expected).is_err());
        assert!(verify_stream_audit(&rows.replace("\t0\t2", "\t0\t0"), &expected).is_err());
        assert!(verify_stream_audit(
            &rows.replace("\t2\t2\t1024\t1024\t0", "\t2\t2\t1025\t1024\t0"),
            &expected,
        ).is_err());
        assert!(verify_stream_audit(
            &rows.replace("\t2\t2\t1024\t1024\t0", "\t2\t2\t1024\t512\t0"),
            &expected,
        ).is_err());
        assert!(verify_stream_audit(
            &rows.replace("\t2\t2\t1024\t0\t2", "\t2\t2\t1024\t1\t2"),
            &expected,
        ).is_err());
        // An all-cached audit cannot prove a cold child ever fetched a slab.
        assert!(verify_stream_audit(
            &rows.replace("\t1024\t1024\t0", "\t1024\t0\t2"),
            &expected
        ).is_err());
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
