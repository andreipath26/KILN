//! Read-only model feasibility preflight for the *currently shipped* runtime.
//! This deliberately separates a size estimate from actual hardware acceptance.
use kiln_models::gguf::GgufFile;
use kiln_runtime::SystemProfile;
use serde_json::{json, Value};

fn mem_available_bytes(contents: &str) -> Option<u64> {
    let line = contents.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() != 3 || fields[2] != "kB" {
        return None;
    }
    fields[1].parse::<u64>().ok()?.checked_mul(1024)
}

/// Parse a kernel cgroup byte counter. The v2 "max" and v1 huge sentinel
/// mean no configured limit, not an available-memory estimate.
fn cgroup_bytes(value: &str) -> Option<u64> {
    let raw = value.trim();
    if raw == "max" || raw.is_empty() || raw.len() > 20 ||
        !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let bytes = raw.parse::<u64>().ok()?;
    if bytes >= (1u64 << 60) { None } else { Some(bytes) }
}

fn cgroup_headroom(limit: &str, current: &str) -> Option<u64> {
    Some(cgroup_bytes(limit)?.saturating_sub(cgroup_bytes(current)?))
}

#[cfg(target_os = "linux")]
fn read_cgroup_number(path: &str) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buffer = [0u8; 64];
    let count = file.read(&mut buffer).ok()?;
    if count == buffer.len() { return None; }
    String::from_utf8(buffer[..count].to_vec()).ok()
}

/// Cgroup v2's memory.max and memory.current represent the container's
/// actual budget and usage. A v1 runtime may instead expose memory.limit_in_bytes.
/// When controls are absent, fall back to the kernel's MemAvailable estimate.
/// The minimum is conservative: page cache may still be reclaimable.
#[cfg(target_os = "linux")]
fn container_memory_headroom() -> Option<u64> {
    for (limit_path, usage_path) in [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ] {
        if let (Some(limit), Some(used)) = (
            read_cgroup_number(limit_path), read_cgroup_number(usage_path),
        ) {
            if let Some(available) = cgroup_headroom(&limit, &used) {
                return Some(available);
            }
        }
    }
    None
}

fn physical_memory_available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let host = std::fs::read_to_string("/proc/meminfo").ok()
            .and_then(|contents| mem_available_bytes(&contents));
        let cgroup = container_memory_headroom();
        match (host, cgroup) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The cgroup CPU quota bounds recommended compute concurrency even when the
/// host advertises more cores. A fractional core is rounded up because the
/// planner still needs at least one runnable worker.
fn cpu_quota_threads(contents: &str) -> Option<u32> {
    let parts: Vec<_> = contents.split_whitespace().collect();
    if parts.len() != 2 || parts[0] == "max" { return None; }
    let quota = parts[0].parse::<u64>().ok()?;
    let period = parts[1].parse::<u64>().ok()?;
    if quota == 0 || period == 0 { return None; }
    let threads = quota / period + u64::from(quota % period != 0);
    Some(threads.clamp(1, u32::MAX as u64) as u32)
}

#[cfg(target_os = "linux")]
fn effective_cpu_quota() -> Option<u32> {
    if let Ok(contents) = std::fs::read_to_string("/sys/fs/cgroup/cpu.max") {
        if let Some(n) = cpu_quota_threads(&contents) { return Some(n); }
    }
    let quota = read_cgroup_number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")?;
    let period = read_cgroup_number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")?;
    cpu_quota_threads(&format!("{quota} {period}"))
}
#[cfg(not(target_os = "linux"))]
fn effective_cpu_quota() -> Option<u32> { None }

fn feasibility(
    model_bytes: u64,
    available_ram_bytes: Option<u64>,
    has_routed_expert_tensors: bool,
    cpu_cores: u32,
    system_ram_gb: u32,
) -> Value {
    // This is NOT the model working-set size. KV, graph scratch, page cache,
    // runtime buffers and other processes are not reliably bounded here.
    let ratio = available_ram_bytes.filter(|bytes| *bytes > 0)
        .map(|bytes| model_bytes as f64 / bytes as f64);
    let status = match (available_ram_bytes, ratio) {
        (Some(0), _) => "model_file_exceeds_current_available_ram",
        (_, Some(value)) if value >= 1.0 => "model_file_exceeds_current_available_ram",
        (_, Some(value)) if value >= 0.7 => "working_set_may_exceed_available_ram",
        (_, Some(_)) => "file_size_within_available_ram_only",
        _ => "memory_feasibility_unknown",
    };
    json!({
        "status": "diagnostic_only_not_load_or_speed_acceptance",
        "model_file_bytes": model_bytes,
        "memory": {
            "current_available_ram_bytes": available_ram_bytes,
            "model_file_to_available_ram_ratio": ratio,
            "system_ram_gb_rounded_from_profile": system_ram_gb,
            "status": status,
            "limitations": "Memory availability is conservatively capped by a detected cgroup limit where possible. File size is not resident working set, KV cache, GPU memory or physical SSD traffic. An oversized GGUF may fail to load or be too slow."
        },
        "cpu": {"logical_cpus_detected": cpu_cores},
        "model": {"routed_expert_tensors_detected": has_routed_expert_tensors},
        "shipping_backend": "llama.cpp-linked development runtime; CPU execution currently verified in CI for the smoke model, accelerator assignment is not auto-validated here",
        "recommendations": {
            "baseline": "Use kiln bench with explicit --threads/--batch/--ubatch and paired token parity",
            "lazy_moe_experts_opt_in": has_routed_expert_tensors,
            "lazy_moe_note": if has_routed_expert_tensors {
                "Experimental --lazy-moe-experts may change mmap/page-cache behavior; it is not application-managed expert streaming and may not improve memory or speed"
            } else {
                "No routed-expert tensor naming evidence; leave lazy MoE disabled"
            },
            "cpu_moe_streamed_end_to_end_ready": false,
            "accelerator_placement_validated": false,
            "measurement_required": "Real GGUF load, exact output parity, peak RSS, SSD I/O and sustained throughput on the target system"
        }
    })
}

pub fn run(model: &str, as_json: bool) -> Result<(), String> {
    // The planner must accept the same local GGUF paths and installed Ollama
    // aliases as 'kiln run' and 'kiln stream'; no model weight copy is needed.
    let resolved = crate::model_library::resolve_model_identifier(model)?;
    let path = resolved.as_path();
    // Inspect *all* GGUF shard headers and tensor ranges. A single first
    // shard is not the model's total file size, expert inventory or evidence
    // that the one-source native expert reader can stream the complete set.
    let split = kiln_models::split_gguf::inspect_split_gguf(path)?;
    let model_data = GgufFile::open(path).map_err(|error| format!(
        "model preflight failed to parse GGUF {}: {error}", path.display()
    ))?;
    let has_routed_experts = model_data.tensors.iter().any(|tensor| {
        tensor.name.contains("_exps") || tensor.name.contains(".expert.")
    }) || model_data.metadata.iter().any(|(key, value)| {
        key.ends_with(".expert_count") && value.as_u32().is_some_and(|count| count > 1)
    });
    // Actual selected-expert CPU MAT_ID graph streaming exists only for
    // GGUFs whose complete fused third-axis layout is verified. Detection
    // must never imply that a large model has passed throughput/quality gates.
    let eligible_fused = kiln_models::discover_complete_fused_moe_selections(&model_data)
        .ok().filter(|items| !items.is_empty());
    let split_native_candidate = split.as_ref().is_some_and(|inventory|
        (2..=64).contains(&inventory.shard_count) &&
        !inventory.fused_expert_slabs.is_empty() &&
        inventory.fused_expert_slabs.len() <= 300_000 &&
        inventory.unsupported_expert_tensors.is_empty()
    );
    let experimental_cpu_external_streaming =
        cfg!(target_os = "linux") &&
        (split_native_candidate || (split.is_none() && eligible_fused.is_some()));
    let total_file_bytes = split.as_ref()
        .map_or(model_data.file_size(), |index| index.total_file_bytes);
    let total_tensors = split.as_ref()
        .map_or(model_data.tensor_count(), |index| index.total_tensor_count);
    let all_shards_have_routed_experts = split.as_ref().is_some_and(|index|
        index.shards.iter().any(|part| part.expert_tensor_count > 0));
    let has_routed_experts = has_routed_experts || all_shards_have_routed_experts;
    let profile = SystemProfile::probe();
    let available = physical_memory_available();
    let mut report = feasibility(
        total_file_bytes, available, has_routed_experts,
        profile.cpu_cores, profile.ram_gb
    );
    report["model_path"] = json!(std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf()).display().to_string());
    report["gguf_tensor_count"] = json!(total_tensors);
    report["split_gguf"] = json!(split.as_ref().map(|index| json!({
        "shard_count": index.shard_count,
        "total_file_bytes": index.total_file_bytes,
        "total_tensor_count": index.total_tensor_count,
        "expert_source_count": index.expert_sources.len(),
        "verified_complete_fused_expert_slab_count": index.fused_expert_slabs.len(),
        "unsupported_expert_tensors": index.unsupported_expert_tensors,
        "shards": index.shards,
        "native_multi_source_streaming_ready": index.native_multi_source_streaming_ready,
        "experimental_native_multi_source_compute_supported": cfg!(target_os="linux") && split_native_candidate,
        "limitation": "Per-shard ranges are verified for supported tensor types, but native multi-file expert computation is NOT implemented"
    })));
    report["architecture"] = json!(model_data.metadata_str("general.architecture"));
    let quota = effective_cpu_quota();
    let recommended_threads = quota
        .map_or(profile.n_threads, |cap| profile.n_threads.min(cap)).max(1);
    report["cpu_threads_suggested"] = json!(recommended_threads);
    report["cpu"]["cgroup_cpu_quota_threads"] = json!(quota);
    report["memory"]["effective_system_ram_gb"] = json!(profile.ram_gb);
    report["profile_tier"] = json!(format!("{:?}", profile.tier));
    report["recommendations"]["cpu_moe_selected_expert_external_compute_experimental"] =
        json!(experimental_cpu_external_streaming);
    report["recommendations"]["verified_fused_expert_tensor_count"] =
        json!(if split.is_some() { 0 } else {
            eligible_fused.as_ref().map(|items| items.len()).unwrap_or(0)
        });
    report["recommendations"]["experimental_source_gguf_index_command"] =
        json!(if split_native_candidate && cfg!(target_os = "linux") {
            "kiln stream <first-shard.gguf> --cache-mib 64 --parallel-reads (experimental native multi-file selected-expert reads)"
        } else if experimental_cpu_external_streaming {
            "kiln-bench expert-export-gguf-stream-index --source-gguf <model.gguf> --auto-verified --output-file <new-index.tsv>"
        } else {
            "not supported for this GGUF layout/platform without explicit expert selection"
        });
    report["recommendations"]["experimental_selected_expert_compute_note"] = json!(
        "Native graph-selected expert pread computation is opt-in via kiln stream. Complete split fused layouts use a verified KILN-MOE-STREAM-2 per-shard index; still experimental and requires measured parity, throughput, RSS and physical SSD evidence"
    );

    if as_json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        println!("KILN read-only model feasibility preflight");
        println!("  model:                {}", report["model_path"].as_str().unwrap_or(model));
        println!("  architecture:         {}", report["architecture"].as_str().unwrap_or("unknown"));
        println!("  model total:          {} bytes", total_file_bytes);
        if let Some(index) = split.as_ref() {
            println!("  split GGUF:           {} source shards, {} tensors, {} expert source ranges", index.shard_count, index.total_tensor_count, index.expert_sources.len());
            println!("  native split stream:  not yet implemented; cannot use single-source index");
        }
        println!("  routed expert tensors: {}", has_routed_experts);
        println!("  logical CPUs:         {}", profile.cpu_cores);
        println!("  available RAM:        {}", available.map(|n| format!("{n} bytes")).unwrap_or_else(|| "unavailable".into()));
        println!("  file/memory finding:  {}", report["memory"]["status"].as_str().unwrap_or("unknown"));
        println!("  proposed CPU threads: {}", recommended_threads);
        println!("  lazy MoE option:      {}", if has_routed_experts { "opt-in experimental only" } else { "not indicated" });
        println!("  external MoE compute:  {}", if experimental_cpu_external_streaming {
            "EXPERIMENTAL native graph-selected expert reads supported by this GGUF layout"
        } else {
            "not automatically supported for this model/platform"
        });
        println!("  external MoE safety:   not deployed; requires opt-in, exact parity, RSS, SSD and throughput gate");
        println!("  accelerator placement: NOT validated by this preflight");
        println!("  IMPORTANT: file size does not predict actual working memory, throughput or accuracy.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_parser_is_strict_and_overflow_safe() {
        assert_eq!(mem_available_bytes("MemTotal: 1000 kB\nMemAvailable: 512 kB\n"), Some(524288));
        assert_eq!(mem_available_bytes("MemAvailable: invalid kB\n"), None);
        assert_eq!(mem_available_bytes("MemAvailable: 2 MiB\n"), None);
        assert_eq!(mem_available_bytes("MemAvailable: 18446744073709551615 kB\n"), None);
    }

    #[test]
    fn cpu_quota_caps_threads_without_guessing_on_malformed_data() {
        assert_eq!(cpu_quota_threads("200000 100000"), Some(2));
        assert_eq!(cpu_quota_threads("120000 100000"), Some(2));
        assert_eq!(cpu_quota_threads("50000 100000"), Some(1));
        assert_eq!(cpu_quota_threads("max 100000"), None);
        assert_eq!(cpu_quota_threads("-1 100000"), None);
        assert_eq!(cpu_quota_threads("0 100000"), None);
        assert_eq!(cpu_quota_threads("100000 0"), None);
        assert_eq!(cpu_quota_threads("100000 100000 extra"), None);
    }

    #[test]
    fn cgroup_budget_is_strict_and_never_wraps() {
        assert_eq!(cgroup_headroom("1073741824\n", "268435456\n"), Some(805306368));
        assert_eq!(cgroup_headroom("268435456", "1073741824"), Some(0));
        assert_eq!(cgroup_headroom("max", "10"), None);
        assert_eq!(cgroup_headroom("9223372036854771712", "10"), None);
        assert_eq!(cgroup_headroom("123bad", "0"), None);
        assert_eq!(cgroup_headroom("100", "not-a-number"), None);
        assert_eq!(cgroup_bytes("18446744073709551616"), None);
    }

    #[test]
    fn oversized_model_never_claims_streaming_deployment() {
        let report = feasibility(12 * 1024 * 1024, Some(4 * 1024 * 1024), true, 8, 16);
        assert_eq!(report["memory"]["status"], "model_file_exceeds_current_available_ram");
        assert_eq!(report["recommendations"]["cpu_moe_streamed_end_to_end_ready"], false);
        assert_eq!(report["recommendations"]["lazy_moe_experts_opt_in"], true);
        let small = feasibility(1024, Some(100000), false, 8, 16);
        assert_eq!(small["memory"]["status"], "file_size_within_available_ram_only");
        assert_eq!(small["recommendations"]["lazy_moe_experts_opt_in"], false);
    }

    #[test]
    fn unknown_memory_never_estimates_passing_hardware_acceptance() {
        let report = feasibility(10000, None, true, 4, 8);
        assert_eq!(report["memory"]["status"], "memory_feasibility_unknown");
        assert!(report["memory"]["model_file_to_available_ram_ratio"].is_null());
        let exhausted = feasibility(1024, Some(0), true, 4, 8);
        assert_eq!(exhausted["memory"]["status"], "model_file_exceeds_current_available_ram");
        assert!(exhausted["memory"]["model_file_to_available_ram_ratio"].is_null());
    }
}
