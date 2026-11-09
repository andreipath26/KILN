//! Read-only model feasibility preflight for the *currently shipped* runtime.
//! This deliberately separates a size estimate from actual hardware acceptance.
use std::path::Path;
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

fn physical_memory_available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo").ok()
            .and_then(|contents| mem_available_bytes(&contents))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn feasibility(
    model_bytes: u64,
    available_ram_bytes: Option<u64>,
    has_routed_expert_tensors: bool,
    cpu_cores: u32,
    system_ram_gb: u32,
) -> Value {
    // This is NOT the model working-set size. KV, graph scratch, page cache,
    // runtime buffers and other processes are not reliably bounded here.
    let ratio = available_ram_bytes.filter(|n| *n > 0)
        .map(|bytes| model_bytes as f64 / bytes as f64);
    let status = match ratio {
        Some(value) if value >= 1.0 => "model_file_exceeds_current_available_ram",
        Some(value) if value >= 0.7 => "working_set_may_exceed_available_ram",
        Some(_) => "file_size_within_available_ram_only",
        None => "memory_feasibility_unknown",
    };
    json!({
        "status": "diagnostic_only_not_load_or_speed_acceptance",
        "model_file_bytes": model_bytes,
        "memory": {
            "current_available_ram_bytes": available_ram_bytes,
            "model_file_to_available_ram_ratio": ratio,
            "system_ram_gb_rounded_from_profile": system_ram_gb,
            "status": status,
            "limitations": "File size is not resident working set, live memory requirement, KV cache, GPU memory or physical SSD traffic. An oversized GGUF may fail to load or be too slow."
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
    let path = Path::new(model);
    let model_data = GgufFile::open(path).map_err(|error| format!(
        "model preflight failed to parse GGUF {}: {error}", path.display()
    ))?;
    let has_routed_experts = model_data.tensors.iter().any(|tensor| {
        tensor.name.contains("_exps") || tensor.name.contains(".expert.")
    }) || model_data.metadata.iter().any(|(key, value)| {
        key.ends_with(".expert_count") && value.as_u32().is_some_and(|count| count > 1)
    });
    let profile = SystemProfile::probe();
    let available = physical_memory_available();
    let mut report = feasibility(
        model_data.file_size(), available, has_routed_experts,
        profile.cpu_cores, profile.ram_gb
    );
    report["model_path"] = json!(std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf()).display().to_string());
    report["gguf_tensor_count"] = json!(model_data.tensor_count());
    report["architecture"] = json!(model_data.metadata_str("general.architecture"));
    report["cpu_threads_suggested"] = json!(profile.n_threads);
    report["profile_tier"] = json!(format!("{:?}", profile.tier));

    if as_json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        println!("KILN read-only model feasibility preflight");
        println!("  model:                {}", report["model_path"].as_str().unwrap_or(model));
        println!("  architecture:         {}", report["architecture"].as_str().unwrap_or("unknown"));
        println!("  model file:           {} bytes", model_data.file_size());
        println!("  routed expert tensors: {}", has_routed_experts);
        println!("  logical CPUs:         {}", profile.cpu_cores);
        println!("  available RAM:        {}", available.map(|n| format!("{n} bytes")).unwrap_or_else(|| "unavailable".into()));
        println!("  file/memory finding:  {}", report["memory"]["status"].as_str().unwrap_or("unknown"));
        println!("  proposed CPU threads: {}", profile.n_threads);
        println!("  lazy MoE option:      {}", if has_routed_experts { "opt-in experimental only" } else { "not indicated" });
        println!("  streamed MoE backend: NOT wired to actual model graph");
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
    }
}
