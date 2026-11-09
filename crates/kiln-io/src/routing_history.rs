//! Model-identity-verified, opt-in hot-expert placement from recorded routes.
//! Historical heat is an ADVISORY ranking, never an authority for actual MoE
//! router decisions or the source of expert bytes.
use std::collections::HashSet;
use std::path::Path;
use kiln_mem::ExpertKey;
use serde_json::Value;
use crate::{CachedExpertBlobReader, ExpertBlobError, ExpertWarmupReport};

const MAX_HISTORY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_WARM_EXPERTS: usize = 65_536;

fn invalid(why: &str) -> ExpertBlobError {
    ExpertBlobError::InvalidRoute(format!("routing history: {why}"))
}
fn get_u64<'a>(object: &'a Value, name: &str) -> Result<u64, ExpertBlobError> {
    object.get(name).and_then(Value::as_u64)
        .ok_or_else(|| invalid("missing or invalid integer field"))
}

/// Read an owner-supplied, explicitly enabled model routing-history artifact.
/// The caller MUST provide a SHA256 calculated from the actual model GGUF,
/// not a digest copied from the history file. Before any disk expert read,
/// all keys, byte lengths, totals, and model identity are checked.
pub fn prewarm_verified_history(
    reader: &CachedExpertBlobReader,
    path: &Path,
    actual_model_sha256: &str,
) -> Result<ExpertWarmupReport, ExpertBlobError> {
    if actual_model_sha256.len() != 64 ||
        !actual_model_sha256.bytes().all(|b| b.is_ascii_hexdigit() &&
            !b.is_ascii_uppercase()) {
        return Err(invalid("actual model SHA256 is not 64 lowercase hex digits"));
    }
    let size = std::fs::metadata(path).map_err(|_| invalid("history unavailable"))?.len();
    if size == 0 || size > MAX_HISTORY_BYTES {
        return Err(invalid("history file exceeds bounded size"));
    }
    let bytes = std::fs::read(path).map_err(|_| invalid("cannot read history"))?;
    if bytes.len() as u64 != size {
        return Err(invalid("history size changed while reading"));
    }
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|_| invalid("invalid history JSON"))?;
    if get_u64(&root, "version")? != 1 ||
        root.get("model_sha256").and_then(Value::as_str) != Some(actual_model_sha256) {
        return Err(invalid("history schema or model identity mismatch"));
    }
    if root.get("runtime_cache_activated") != Some(&Value::Bool(false)) {
        return Err(invalid("history unexpectedly claims live cache activation"));
    }
    let admission = root.get("admission").ok_or_else(|| invalid("missing admission plan"))?;
    if admission.get("actual_routing_modified") != Some(&Value::Bool(false)) ||
        admission.get("runtime_cache_activated") != Some(&Value::Bool(false)) {
        return Err(invalid("admission must preserve actual model routing"));
    }
    let budget = get_u64(admission, "cache_budget_bytes")?;
    let selected_bytes = get_u64(admission, "selected_bytes")?;
    if budget > (1u64 << 50) || selected_bytes > budget {
        return Err(invalid("admission byte budget is inconsistent"));
    }
    let rows = admission.get("selected_experts")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing selected expert list"))?;
    if rows.len() > MAX_WARM_EXPERTS {
        return Err(invalid("too many selected experts"));
    }
    let mut unique = HashSet::with_capacity(rows.len());
    let mut ranked = Vec::with_capacity(rows.len());
    let mut total = 0u64;
    for row in rows {
        let layer = u32::try_from(get_u64(row, "layer")?)
            .map_err(|_| invalid("layer ID out of range"))?;
        let expert_id = u32::try_from(get_u64(row, "expert_id")?)
            .map_err(|_| invalid("expert ID out of range"))?;
        let key = ExpertKey { layer, expert_id };
        if !unique.insert(key) {
            return Err(invalid("duplicate selected expert"));
        }
        if get_u64(row, "score")? == 0 {
            return Err(invalid("zero-heat expert in selected plan"));
        }
        let claimed = get_u64(row, "bytes")?;
        let actual = reader.expert_byte_length(layer, expert_id)?;
        if claimed == 0 || claimed != actual {
            return Err(invalid("history expert bytes disagree with verified source index"));
        }
        total = total.checked_add(actual)
            .ok_or_else(|| invalid("selected expert byte overflow"))?;
        if total > budget {
            return Err(invalid("selected experts exceed budget"));
        }
        ranked.push(key);
    }
    if total != selected_bytes {
        return Err(invalid("selected byte total does not match verified index"));
    }
    reader.prewarm_ranked(&ranked, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::{ExpertRange, ExpertTable};
    use serde_json::json;

    #[test]
    fn verified_hot_history_is_opt_in_and_never_overrides_router() {
        let root = std::env::temp_dir().join(format!(
            "kiln-hot-history-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let data = root.join("experts.bin");
        let history = root.join("history.json");
        std::fs::write(&data, [9u8; 32]).unwrap();
        let table = ExpertTable {blob_version:1,blob_size:32,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
            ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
        ]};
        let reader = CachedExpertBlobReader::open(
            &data, table, 16, ExpertCachePolicy::Adaptive
        ).unwrap();
        let sha = "a".repeat(64);
        let mut report = json!({
            "version":1, "model_sha256":sha,
            "runtime_cache_activated":false,
            "admission":{
                "actual_routing_modified":false,"runtime_cache_activated":false,
                "cache_budget_bytes":16,"selected_bytes":16,
                "selected_experts":[{"layer":0,"expert_id":1,"score":5,"bytes":16}]
            }
        });
        let write = |value: &Value| {
            std::fs::write(&history, serde_json::to_vec(value).unwrap()).unwrap();
        };
        write(&report);
        assert_eq!(reader.cache_stats().entries, 0);
        let ready = prewarm_verified_history(&reader, &history, &sha).unwrap();
        assert_eq!(ready.admitted_bytes, 16);
        assert_eq!(reader.cache_stats().hits, 0);
        assert_eq!(reader.read_expert(0, 0).unwrap(), &[9u8;16]);
        assert_eq!(reader.cache_stats().used_bytes, 16);
        report["admission"]["selected_experts"][0]["bytes"] = json!(15);
        write(&report);
        let misses = reader.cache_stats().misses;
        assert!(prewarm_verified_history(&reader, &history, &sha).is_err());
        assert_eq!(reader.cache_stats().misses, misses);
        report["admission"]["selected_experts"][0]["bytes"] = json!(16);
        write(&report);
        assert!(prewarm_verified_history(&reader, &history, &"b".repeat(64)).is_err());
        report["admission"]["selected_experts"].as_array_mut().unwrap()
            .push(json!({"layer":0,"expert_id":1,"score":5,"bytes":16}));
        report["admission"]["selected_bytes"] = json!(32);
        report["admission"]["cache_budget_bytes"] = json!(32);
        write(&report);
        assert!(prewarm_verified_history(&reader, &history, &sha).is_err());
        std::fs::remove_file(history).unwrap();
        std::fs::remove_file(data).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
