//! KILN benchmark harness.
//!
//! Measures the runtime and produces a JSON report. In Phase 0B it
//! measures the infrastructure: hardware detection, selector planning,
//! scheduler stepping, and process memory. Model throughput measurement
//! lands in Phase 1 when the runtime can actually run models.
//!
//! Usage:
//!   kiln-bench run            # run all benchmarks and print report
//!   kiln-bench run --json     # print report as JSON only
//!   kiln-bench info           # print what will be measured

use std::collections::HashMap;
use std::time::Instant;
use std::path::PathBuf;

use kiln_io::{write_expert_sidecar, verify_expert_sidecar, write_component_expert_sidecar, verify_component_expert_sidecar, CachedExpertBlobReader, FileRangeReader};
use kiln_mem::{evaluate_grouped_router_holdout, RouterEvent, evaluate_predictor_holdout, ExpertCachePolicy, ExpertKey, ExpertPrefetcher, PrefetchStats};
use kiln_models::{ExpertRange, ExpertTable, NamedExpertTensor, resolve_expert_tensors_in_gguf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use kiln_core::{LinuxMonitor, PerformanceMonitor};

/// The benchmark harness.
#[derive(Parser, Debug)]
#[command(name = "kiln-bench", version, about = "KILN benchmark harness")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the full benchmark suite and print the report.
    Run {
        /// Print the report as JSON only, no human-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Print what will be measured without running anything.
    Info,
    /// Compare real-file expert reads with no cache and byte-bounded cache policies.
    ExpertCache {
        /// Path to an expert blob or representative large file.
        #[arg(long)]
        file: PathBuf,
        /// Bytes per synthetic expert block (default 1 MiB).
        #[arg(long, default_value_t = 1_048_576)]
        block_bytes: usize,
        /// Number of expert accesses to replay.
        #[arg(long, default_value_t = 10_000)]
        requests: usize,
        /// Equal retained-cache budget for LRU, LFU and adaptive policies.
        #[arg(long, default_value_t = 256)]
        cache_mib: u64,
        /// Seed for reproducible access traces.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Synthetic access pattern; ignored when --trace-file is supplied.
        #[arg(long, value_parser = ["phase-shift", "stationary-hot", "uniform", "round-robin"], default_value = "phase-shift")]
        workload: String,
        /// Optional validated JSON ExpertTable mapping (layer, expert_id) to blob byte ranges.
        #[arg(long)]
        index_file: Option<PathBuf>,
        /// Optional JSON array of expert IDs or layer-aware keys captured from a model router.
        #[arg(long)]
        trace_file: Option<PathBuf>,
        /// Emit JSON only.
        #[arg(long)]
        json: bool,
    },
    /// Replay a mapped expert-ID trace through the bounded history prefetch predictor.
    PrefetchReplay {
        /// JSON array of expert IDs or layer-aware expert keys in observed access order.
        #[arg(long)]
        trace_file: PathBuf,
        /// Optional expert blob file; required together with --index-file for variable-size accounting.
        #[arg(long)]
        blob_file: Option<PathBuf>,
        /// Optional JSON ExpertTable index; when supplied, candidate byte sizes come from indexed ranges.
        #[arg(long)]
        index_file: Option<PathBuf>,
        /// Fixed expert block size used for planner budget accounting without --index-file.
        #[arg(long, default_value_t = 1_048_576)]
        block_bytes: u64,
        /// Maximum number of candidate experts to schedule per observed access.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// Maximum total tracked plans (in-flight plus ready).
        #[arg(long, default_value_t = 8)]
        max_pending: usize,
        /// Maximum in-flight prefetch bytes.
        #[arg(long, default_value_t = 16)]
        max_inflight_mib: u64,
        /// Simulated read latency in trace-access steps; 0 means ready immediately.
        #[arg(long, default_value_t = 1)]
        latency_requests: usize,
        /// Emit JSON only.
        #[arg(long)]
        json: bool,
    },
    /// Holdout-only next-expert prediction accuracy, with no training leakage.
    PredictorHoldout {
        /// JSON array of expert IDs, (layer,expert_id) pairs or expert-key objects.
        #[arg(long)]
        trace_file: PathBuf,
        /// Number of chronological events to reserve for training.
        #[arg(long)]
        train_events: usize,
        /// Number of candidate next experts: Phase 5.0 target is k=2.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        #[arg(long)]
        json: bool,
    },
    /// Grouped MoE router-event next-expert holdout evaluation (top-k expert set).
    RouterHoldout {
        /// Chronological JSON array of {token_position,layer,active_experts}.
        #[arg(long)]
        trace_file: PathBuf,
        /// All tokens before this index train the transition model.
        #[arg(long)]
        train_until_token: u64,
        /// Number of simultaneous expert candidates.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        #[arg(long)]
        json: bool,
    },
    /// Pack explicitly indexed expert byte ranges into a safe GGUF-preserving sidecar.
    ExpertPack {
        #[arg(long)]
        source_file: PathBuf,
        /// JSON ExpertTable of source ranges, validated against source length.
        #[arg(long)]
        source_index: PathBuf,
        /// New file; will never overwrite existing output.
        #[arg(long)]
        blob_file: PathBuf,
        /// New JSON ExpertTable containing aligned sidecar offsets.
        #[arg(long)]
        index_file: PathBuf,
        /// New integrity JSON with per-expert CRC32 values.
        #[arg(long)]
        integrity_file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Pack exact named GGUF tensors grouped by expert into an aligned sidecar.
    ExpertPackNamed {
        /// Intact GGUF, opened read-only.
        #[arg(long)]
        source_gguf: PathBuf,
        /// JSON list of {layer,expert_id,tensor_name}, explicitly model-aware.
        #[arg(long)]
        selections_file: PathBuf,
        /// Create-new expert blob; will not overwrite another file.
        #[arg(long)]
        blob_file: PathBuf,
        /// Create-new JSON ExpertTable index.
        #[arg(long)]
        index_file: PathBuf,
        /// Create-new per-expert CRC32 metadata.
        #[arg(long)]
        integrity_file: PathBuf,
        /// Create-new component boundary manifest.
        #[arg(long)]
        components_file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Validate full sidecar index and all expert payload CRC32 values.
    ExpertPackVerify {
        #[arg(long)]
        blob_file: PathBuf,
        #[arg(long)]
        index_file: PathBuf,
        #[arg(long)]
        integrity_file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Measure bounded random and sequential reads from a real model/expert blob.
    ExpertIo {
        /// Path to a model file or expert sidecar containing indexed expert ranges.
        #[arg(long)]
        file: PathBuf,
        /// Validated ExpertTable mapping (layer, expert_id) to real file ranges.
        #[arg(long)]
        index_file: Option<PathBuf>,
        /// Replay chronological real-router expert IDs using the validated index.
        /// Accepts the same JSON formats as prefetch-replay; requires --index-file.
        #[arg(long)]
        trace_file: Option<PathBuf>,
        /// Bytes per expert-block read (default 1 MiB).
        #[arg(long, default_value_t = 1_048_576)]
        block_bytes: usize,
        /// Number of block reads per repetition.
        #[arg(long, default_value_t = 256)]
        requests: usize,
        /// Number of repetitions for each access pattern.
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Seed for reproducible random access offsets.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Advise Linux to evict this file's cached pages before each access pattern.
        /// This is best effort, never a guaranteed cold-device measurement.
        #[arg(long)]
        advise_drop_cache: bool,
        /// Emit JSON only.
        #[arg(long)]
        json: bool,
    },
}

/// The report structure. This is the format every phase gate produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub version: String,
    pub phase: String,
    pub started_at_unix: u64,
    pub measurements: Vec<Measurement>,
    pub summary: Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub success: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertIoPattern {
    access_pattern: String,
    requests: usize,
    block_bytes: usize,
    min_request_bytes: usize,
    max_request_bytes: usize,
    total_bytes_read: u64,
    /// Expert identities sampled in request order for one repetition.
    expert_keys: Vec<ExpertKey>,
    /// Linux /proc/self/io read_bytes delta for this pattern, when accessible.
    /// Includes read-ahead and cannot be equated with device-only bytes.
    linux_process_storage_read_bytes: Option<u64>,
    /// Linux file-scoped eviction hint, not verified physical cache residency.
    page_cache_eviction_advised: bool,
    elapsed_ms: f64,
    throughput_mib_s: f64,
    request_latency_p50_us: f64,
    request_latency_p95_us: f64,
    checksum: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertCachePattern {
    policy: String,
    cache_capacity_bytes: u64,
    requests: usize,
    hits: u64,
    misses: u64,
    hit_rate: f64,
    estimated_backing_bytes_read: u64,
    elapsed_ms: f64,
    requests_per_second: f64,
    cache_used_bytes: u64,
    entries: usize,
    evictions: u64,
    checksum: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertCacheReport {
    version: String,
    phase_gate: String,
    file: String,
    file_size_bytes: u64,
    block_bytes: Option<usize>,
    index_file: Option<String>,
    expert_entries_available: usize,
    requests: usize,
    seed: u64,
    workload: String,
    trace_source: String,
    page_cache_controlled: bool,
    note: String,
    patterns: Vec<ExpertCachePattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefetchReplayReport {
    version: String,
    phase_gate: String,
    trace_file: String,
    blob_file: Option<String>,
    index_file: Option<String>,
    trace_events: usize,
    distinct_experts_in_trace: usize,
    block_bytes: Option<u64>,
    candidate_bytes_total: u64,
    top_k: usize,
    max_pending: usize,
    max_inflight_bytes: u64,
    simulated_latency_requests: usize,
    next_event_coverage: f64,
    next_event_precision: f64,
    predicted_candidates: u64,
    correct_next_event_predictions: u64,
    stats: PrefetchStats,
    note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertIoReport {
    version: String,
    phase_gate: String,
    file: String,
    file_size_bytes: u64,
    index_file: Option<String>,
    indexed_expert_ranges: Option<usize>,
    os: String,
    architecture: String,
    block_bytes: usize,
    requests_per_run: usize,
    repetitions: usize,
    seed: u64,
    page_cache_controlled: bool,
    note: String,
    patterns: Vec<ExpertIoPattern>,
}

fn percentile(sorted: &[u64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index.min(sorted.len() - 1)] as f64 / 1_000.0
}

/// Parse the Linux process read_bytes counter. Not the sum of read() calls:
/// it represents storage-backed bytes attributed to this process by the kernel.
/// This is diagnostic only; page cache and filesystem effects still apply.
fn parse_linux_read_bytes(contents: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim() == "read_bytes" {
            value.trim().parse::<u64>().ok()
        } else {
            None
        }
    })
}

fn linux_process_read_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/io")
            .ok()
            .and_then(|text| parse_linux_read_bytes(&text))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod physical_io_counter_tests {
    use super::parse_linux_read_bytes;

    #[test]
    fn parses_only_real_read_bytes_key() {
        let contents = "rchar: 100\nwchar: 20\nsyscr: 5\nread_bytes: 8192\nwrite_bytes: 2\n";
        assert_eq!(parse_linux_read_bytes(contents), Some(8192));
        assert_eq!(parse_linux_read_bytes("rchar: 123\n"), None);
        assert_eq!(parse_linux_read_bytes("read_bytes: nope\n"), None);
        assert_eq!(parse_linux_read_bytes("not_read_bytes: 1\n"), None);
    }
}

#[derive(Debug, Clone)]
struct ExpertIoRequest {
    offset: u64,
    length: usize,
    expert_key: Option<ExpertKey>,
}

fn run_expert_io_pattern(
    reader: &mut FileRangeReader,
    requests: &[ExpertIoRequest],
    block_bytes: usize,
    repetitions: usize,
    access_pattern: &str,
) -> Result<ExpertIoPattern, Box<dyn std::error::Error>> {
    run_expert_io_pattern_with_advice(reader, requests, block_bytes, repetitions, access_pattern, false)
}

fn run_expert_io_pattern_with_advice(
    reader: &mut FileRangeReader,
    requests: &[ExpertIoRequest],
    block_bytes: usize,
    repetitions: usize,
    access_pattern: &str,
    advise_drop_cache: bool,
) -> Result<ExpertIoPattern, Box<dyn std::error::Error>> {
    if requests.is_empty() || requests.iter().any(|request| request.length == 0 || request.length > block_bytes) {
        return Err("I/O request list must be non-empty and every request length must be within 1..=block-bytes".into());
    }
    if advise_drop_cache {
        // Do not claim successful cold-cache control on unsupported systems.
        reader.advise_drop_cache()?;
    }
    let mut buffer = vec![0u8; block_bytes];
    let measured_requests = requests.len().saturating_mul(repetitions);
    let mut latencies = Vec::with_capacity(measured_requests);
    let mut checksum = 0u64;
    let mut expert_keys = Vec::with_capacity(requests.len());
    let storage_before = linux_process_read_bytes();
    let total_start = Instant::now();

    for repetition in 0..repetitions {
        for request in requests {
            let start = Instant::now();
            let bytes = &mut buffer[..request.length];
            reader.read_exact_at(request.offset, bytes)?;
            latencies.push(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            if repetition == 0 {
                if let Some(key) = request.expert_key {
                    expert_keys.push(key);
                }
            }
            // Touch a deterministic sample of the bytes to prevent this from
            // becoming a pure seek-only benchmark.
            for byte in bytes.iter().step_by(4096) {
                checksum = checksum.rotate_left(5) ^ u64::from(*byte);
            }
        }
    }

    let elapsed = total_start.elapsed();
    let storage_after = linux_process_read_bytes();
    let linux_process_storage_read_bytes = storage_before.zip(storage_after)
        .and_then(|(before, after)| after.checked_sub(before));
    let elapsed_secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let bytes_per_run = requests.iter().fold(0u64, |sum, request| {
        sum.saturating_add(request.length as u64)
    });
    let total_bytes_read = bytes_per_run.saturating_mul(repetitions as u64);
    let min_request_bytes = requests.iter().map(|request| request.length).min().unwrap_or(0);
    let max_request_bytes = requests.iter().map(|request| request.length).max().unwrap_or(0);
    latencies.sort_unstable();

    Ok(ExpertIoPattern {
        access_pattern: access_pattern.to_string(),
        requests: measured_requests,
        block_bytes,
        min_request_bytes,
        max_request_bytes,
        total_bytes_read,
        expert_keys,
        linux_process_storage_read_bytes,
        page_cache_eviction_advised: advise_drop_cache,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_mib_s: total_bytes_read as f64 / (1024.0 * 1024.0) / elapsed_secs,
        request_latency_p50_us: percentile(&latencies, 0.50),
        request_latency_p95_us: percentile(&latencies, 0.95),
        checksum,
    })
}

fn run_expert_io(
    file: &std::path::Path,
    block_bytes: usize,
    requests: usize,
    runs: usize,
    seed: u64,
) -> Result<ExpertIoReport, Box<dyn std::error::Error>> {
    validate_expert_io_args(block_bytes, requests, runs)?;
    let mut reader = FileRangeReader::open(file)?;
    let blocks_available = reader.file_size() / block_bytes as u64;
    if blocks_available == 0 {
        return Err(format!(
            "file is {} bytes; it must contain at least one full {}-byte block",
            reader.file_size(), block_bytes
        ).into());
    }

    let mut sequential_requests = Vec::with_capacity(requests);
    let mut random_requests = Vec::with_capacity(requests);
    let mut state = seed;
    for i in 0..requests {
        sequential_requests.push(ExpertIoRequest {
            offset: (i as u64 % blocks_available) * block_bytes as u64,
            length: block_bytes,
            expert_key: None,
        });
        // Stable LCG: repeatable across platforms and Rust versions.
        state = next_expert_io_random(state);
        random_requests.push(ExpertIoRequest {
            offset: (state % blocks_available) * block_bytes as u64,
            length: block_bytes,
            expert_key: None,
        });
    }

    let sequential = run_expert_io_pattern(&mut reader, &sequential_requests, block_bytes, runs, "sequential")?;
    let random = run_expert_io_pattern(&mut reader, &random_requests, block_bytes, runs, "random")?;

    Ok(expert_io_report(file, reader.file_size(), None, None, block_bytes, requests, runs, seed,
        "Fixed-size file offsets are a synthetic layout, not mapped MoE expert ranges.".to_string(),
        vec![sequential, random]))
}

fn validate_expert_io_args(
    block_bytes: usize,
    requests: usize,
    runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if block_bytes == 0 || requests == 0 || runs == 0 {
        return Err("block-bytes, requests, and runs must all be greater than zero".into());
    }
    if block_bytes > 64 * 1024 * 1024 || requests > 100_000 || runs > 100
        || requests.checked_mul(runs).filter(|&n| n <= 1_000_000).is_none() {
        return Err("expert-io limits are block-bytes <= 64 MiB, requests <= 100,000, runs <= 100, and requests*runs <= 1,000,000".into());
    }
    Ok(())
}

fn next_expert_io_random(state: u64) -> u64 {
    state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407)
}

fn expert_io_report(
    file: &std::path::Path,
    file_size_bytes: u64,
    index_file: Option<&std::path::Path>,
    indexed_expert_ranges: Option<usize>,
    block_bytes: usize,
    requests: usize,
    runs: usize,
    seed: u64,
    range_note: String,
    patterns: Vec<ExpertIoPattern>,
) -> ExpertIoReport {
    ExpertIoReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase_gate: "5A-expert-block-io-feasibility".to_string(),
        file: file.display().to_string(),
        file_size_bytes,
        index_file: index_file.map(|path| path.display().to_string()),
        indexed_expert_ranges,
        os: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        block_bytes,
        requests_per_run: requests,
        repetitions: runs,
        seed,
        page_cache_controlled: false,
        note: format!("{range_note} OS page cache is uncontrolled; repeated runs may be warm. Linux /proc/self/io read_bytes is process-attributed and includes readahead; it is not a direct device-byte counter. This probe does not measure model inference tokens/s."),
        patterns,
    }
}

fn run_expert_io_indexed(
    file: &std::path::Path,
    index_file: &std::path::Path,
    block_bytes: usize,
    requests: usize,
    runs: usize,
    seed: u64,
) -> Result<ExpertIoReport, Box<dyn std::error::Error>> {
    run_expert_io_indexed_with_advice(file, index_file, block_bytes, requests, runs, seed, false)
}

fn run_expert_io_indexed_with_advice(
    file: &std::path::Path,
    index_file: &std::path::Path,
    block_bytes: usize,
    requests: usize,
    runs: usize,
    seed: u64,
    advise_drop_cache: bool,
) -> Result<ExpertIoReport, Box<dyn std::error::Error>> {
    validate_expert_io_args(block_bytes, requests, runs)?;
    let mut reader = FileRangeReader::open(file)?;
    let index_bytes = read_bounded_input(index_file, MAX_EXPERT_INDEX_BYTES, "expert range index")?;
    let table: ExpertTable = serde_json::from_slice(&index_bytes)?;
    table.validate(reader.file_size())?;
    if table.ranges.is_empty() {
        return Err("expert range index contains no entries".into());
    }
    let mut ranges: Vec<_> = table.ranges.iter().collect();
    ranges.sort_by_key(|range| range.byte_offset);

    let mut sequential = Vec::with_capacity(requests);
    let mut random = Vec::with_capacity(requests);
    let mut state = seed;
    for i in 0..requests {
        // Spread sequential samples across the full offset-sorted expert index,
        // rather than accidentally timing only the first layers.
        let selected = ((i as u128 * ranges.len() as u128) / requests as u128) as usize;
        let range = ranges[selected.min(ranges.len() - 1)];
        let sequential_length = usize::try_from(range.byte_length.min(block_bytes as u64))?;
        sequential.push(ExpertIoRequest {
            offset: range.byte_offset,
            length: sequential_length,
            expert_key: Some(ExpertKey { layer: range.layer, expert_id: range.expert_id }),
        });

        state = next_expert_io_random(state);
        let random_range = ranges[(state % ranges.len() as u64) as usize];
        let random_length = usize::try_from(random_range.byte_length.min(block_bytes as u64))?;
        let available_starts = random_range.byte_length - random_length as u64 + 1;
        state = next_expert_io_random(state);
        let within_expert = state % available_starts;
        random.push(ExpertIoRequest {
            offset: random_range.byte_offset + within_expert,
            length: random_length,
            expert_key: Some(ExpertKey { layer: random_range.layer, expert_id: random_range.expert_id }),
        });
    }

    let sequential_pattern = run_expert_io_pattern_with_advice(&mut reader, &sequential, block_bytes, runs, "sequential", advise_drop_cache)?;
    let random_pattern = run_expert_io_pattern_with_advice(&mut reader, &random, block_bytes, runs, "random", advise_drop_cache)?;
    Ok(expert_io_report(
        file,
        reader.file_size(),
        Some(index_file),
        Some(table.ranges.len()),
        block_bytes,
        requests,
        runs,
        seed,
        format!("Reads are sampled from validated ExpertTable ranges; sequential samples follow file-offset order and random samples choose offsets within real experts. {} indexed expert ranges were available.", table.ranges.len()),
        vec![sequential_pattern, random_pattern],
    ))
}

/// Replay real router-selected experts in their original order through bounded
/// buffered file reads. Unlike synthetic sampling, this preserves repeated hot
/// experts and cross-layer transitions. It does not run model inference.
fn run_expert_io_trace(
    file: &std::path::Path,
    index_file: &std::path::Path,
    trace_file: &std::path::Path,
    block_bytes: usize,
    runs: usize,
) -> Result<ExpertIoReport, Box<dyn std::error::Error>> {
    run_expert_io_trace_with_advice(file, index_file, trace_file, block_bytes, runs, false)
}

fn run_expert_io_trace_with_advice(
    file: &std::path::Path,
    index_file: &std::path::Path,
    trace_file: &std::path::Path,
    block_bytes: usize,
    runs: usize,
    advise_drop_cache: bool,
) -> Result<ExpertIoReport, Box<dyn std::error::Error>> {
    validate_expert_io_args(block_bytes, 1, runs)?;
    let mut reader = FileRangeReader::open(file)?;
    let index_bytes = read_bounded_input(index_file, MAX_EXPERT_INDEX_BYTES, "expert range index")?;
    let table: ExpertTable = serde_json::from_slice(&index_bytes)?;
    table.validate(reader.file_size())?;
    if table.ranges.is_empty() {
        return Err("expert range index contains no entries".into());
    }
    let input: ExpertTraceInput = serde_json::from_slice(
        &read_bounded_input(trace_file, MAX_EXPERT_TRACE_BYTES, "expert trace")?
    )?;
    let trace: Vec<ExpertKey> = match input {
        ExpertTraceInput::FlatIds(ids) => ids.into_iter()
            .map(|expert_id| ExpertKey { layer: 0, expert_id }).collect(),
        ExpertTraceInput::ExpertKeys(keys) => keys,
        ExpertTraceInput::Pairs(pairs) => pairs.into_iter()
            .map(|(layer, expert_id)| ExpertKey { layer, expert_id }).collect(),
    };
    if trace.is_empty() || trace.len() > 100_000
        || trace.len().checked_mul(runs).filter(|&n| n <= 1_000_000).is_none() {
        return Err("trace replay requires 1..=100,000 expert events and at most 1,000,000 total reads".into());
    }
    let indexed: HashMap<ExpertKey, &ExpertRange> = table.ranges.iter()
        .map(|range| (ExpertKey { layer: range.layer, expert_id: range.expert_id }, range))
        .collect();
    let mut mapped = Vec::with_capacity(trace.len());
    for key in trace {
        let range = indexed.get(&key).ok_or_else(|| format!(
            "router trace references missing expert (layer {}, expert {})", key.layer, key.expert_id
        ))?;
        mapped.push(ExpertIoRequest {
            offset: range.byte_offset,
            length: usize::try_from(range.byte_length.min(block_bytes as u64))?,
            expert_key: Some(key),
        });
    }
    let pattern = run_expert_io_pattern_with_advice(&mut reader, &mapped, block_bytes, runs, "router-trace", advise_drop_cache)?;
    Ok(expert_io_report(
        file, reader.file_size(), Some(index_file), Some(table.ranges.len()),
        block_bytes, mapped.len(), runs, 0,
        format!("Chronological router trace replay from {}; exact mapped expert start offsets and up to {} bytes per selected expert. Does not reproduce model compute or include unselected fused-expert components.", trace_file.display(), block_bytes),
        vec![pattern],
    ))
}

fn make_key_trace(
    available: &[ExpertKey],
    requests: usize,
    seed: u64,
    workload: &str,
) -> Vec<ExpertKey> {
    let mut trace = Vec::with_capacity(requests);
    if available.is_empty() {
        return trace;
    }
    let mut state = seed;
    let hot_count = (available.len() / 8).max(1);
    let next = |state: &mut u64| {
        *state = state.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    };
    for i in 0..requests {
        let index = match workload {
            "uniform" => (next(&mut state) % available.len() as u64) as usize,
            "round-robin" => i % available.len(),
            "stationary-hot" | "phase-shift" => {
                let phase = if workload == "stationary-hot" || i < requests / 2 { 0 } else { 1 };
                let hot_start = phase * (available.len() / 2);
                if next(&mut state) % 100 < 85 {
                    hot_start + (next(&mut state) % hot_count as u64) as usize
                } else {
                    (next(&mut state) % available.len() as u64) as usize
                }
            }
            _ => unreachable!("workload validated by CLI and caller"),
        };
        trace.push(available[index]);
    }
    trace
}

fn run_expert_cache_policy(
    file: &std::path::Path,
    table: ExpertTable,
    trace: &[ExpertKey],
    cache_capacity_bytes: u64,
    policy: ExpertCachePolicy,
    policy_name: &str,
) -> Result<ExpertCachePattern, Box<dyn std::error::Error>> {
    let reader = CachedExpertBlobReader::open(file, table, cache_capacity_bytes, policy)?;
    let started = Instant::now();
    let mut checksum = 0u64;
    let mut estimated_backing_bytes_read = 0u64;
    let mut previous_misses = 0u64;
    for key in trace {
        let bytes = reader.read_expert(key.layer, key.expert_id)?;
        let stats = reader.cache_stats();
        if stats.misses > previous_misses {
            estimated_backing_bytes_read = estimated_backing_bytes_read.saturating_add(bytes.len() as u64);
            previous_misses = stats.misses;
        }
        if let Some(first) = bytes.first() {
            checksum = checksum.rotate_left(5) ^ u64::from(*first);
        }
        if let Some(last) = bytes.last() {
            checksum = checksum.rotate_left(3) ^ u64::from(*last);
        }
    }
    let elapsed = started.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let stats = reader.cache_stats();
    let requests = trace.len();
    let total_lookups = stats.hits.saturating_add(stats.misses).max(1);
    Ok(ExpertCachePattern {
        policy: policy_name.to_string(),
        cache_capacity_bytes,
        requests,
        hits: stats.hits,
        misses: stats.misses,
        hit_rate: stats.hits as f64 / total_lookups as f64,
        estimated_backing_bytes_read,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        requests_per_second: requests as f64 / elapsed_secs,
        cache_used_bytes: stats.used_bytes,
        entries: stats.entries,
        evictions: stats.evictions,
        checksum,
    })
}

fn run_expert_cache(
    file: &std::path::Path,
    block_bytes: usize,
    requests: usize,
    cache_mib: u64,
    seed: u64,
    workload: &str,
    trace_file: Option<&std::path::Path>,
    index_file: Option<&std::path::Path>,
) -> Result<ExpertCacheReport, Box<dyn std::error::Error>> {
    if (trace_file.is_none() && requests == 0) || cache_mib == 0 {
        return Err("requests must be greater than zero without --trace-file, and cache-mib must be greater than zero".into());
    }
    if trace_file.is_none() && !["phase-shift", "stationary-hot", "uniform", "round-robin"].contains(&workload) {
        return Err(format!("unsupported workload: {workload}").into());
    }
    let file_size = std::fs::metadata(file)?.len();
    let (table, block_bytes_report) = if let Some(index_path) = index_file {
        let contents = read_bounded_input(index_path, MAX_EXPERT_INDEX_BYTES, "expert index")?;
        let table: ExpertTable = serde_json::from_slice(&contents)?;
        table.validate(file_size)?;
        if table.ranges.is_empty() {
            return Err("expert index contains no entries".into());
        }
        (table, None)
    } else {
        if block_bytes == 0 {
            return Err("block-bytes must be greater than zero when --index-file is not supplied".into());
        }
        let block_bytes_u64 = u64::try_from(block_bytes)?;
        let experts_u64 = file_size / block_bytes_u64;
        if experts_u64 == 0 {
            return Err(format!("file is {file_size} bytes; it must contain at least one full {block_bytes}-byte block").into());
        }
        if experts_u64 > 1_000_000 {
            return Err("benchmark index would exceed one million synthetic expert blocks; increase --block-bytes".into());
        }
        let mut ranges = Vec::with_capacity(experts_u64 as usize);
        for expert_id in 0..experts_u64 {
            ranges.push(ExpertRange {
                layer: 0,
                expert_id: expert_id as u32,
                byte_offset: expert_id.saturating_mul(block_bytes_u64),
                byte_length: block_bytes_u64,
            });
        }
        (ExpertTable { blob_version: 1, blob_size: file_size, ranges }, Some(block_bytes))
    };
    table.validate(file_size)?;
    let available: Vec<ExpertKey> = table.ranges.iter()
        .map(|range| ExpertKey { layer: range.layer, expert_id: range.expert_id })
        .collect();
    if available.len() > 1_000_000 {
        return Err("expert index contains more than one million entries; split or reduce the index to bound benchmark memory".into());
    }
    let trace: Vec<ExpertKey> = if let Some(trace_path) = trace_file {
        let input: ExpertTraceInput = serde_json::from_slice(&read_bounded_input(trace_path, MAX_EXPERT_TRACE_BYTES, "expert trace")?)?;
        let trace = match input {
            ExpertTraceInput::FlatIds(ids) => ids.into_iter()
                .map(|expert_id| ExpertKey { layer: 0, expert_id }).collect(),
            ExpertTraceInput::ExpertKeys(keys) => keys,
            ExpertTraceInput::Pairs(pairs) => pairs.into_iter()
                .map(|(layer, expert_id)| ExpertKey { layer, expert_id }).collect(),
        };
        if trace.is_empty() {
            return Err("trace file must be a non-empty JSON array of expert IDs, [layer, expert_id] pairs, or {layer, expert_id} objects".into());
        }
        if trace.len() > 10_000_000 {
            return Err("trace contains more than 10 million events; split the trace to bound memory".into());
        }
        let available_set: std::collections::HashSet<ExpertKey> = available.iter().copied().collect();
        if let Some(missing) = trace.iter().find(|key| !available_set.contains(key)) {
            return Err(format!("trace references expert ({}, {}), which is absent from the selected index", missing.layer, missing.expert_id).into());
        }
        trace
    } else {
        make_key_trace(&available, requests, seed, workload)
    };
    let cache_capacity_bytes = cache_mib.checked_mul(1024 * 1024)
        .ok_or("cache-mib is too large")?;
    let mut patterns = Vec::new();
    patterns.push(run_expert_cache_policy(
        file, table.clone(), &trace, 0, ExpertCachePolicy::Lru, "none"
    )?);
    for (policy, name) in [
        (ExpertCachePolicy::Lru, "lru"),
        (ExpertCachePolicy::Lfu, "lfu"),
        (ExpertCachePolicy::Adaptive, "adaptive"),
    ] {
        patterns.push(run_expert_cache_policy(
            file, table.clone(), &trace, cache_capacity_bytes, policy, name
        )?);
    }
    Ok(ExpertCacheReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase_gate: "5B-expert-cache-trace-replay".to_string(),
        file: file.display().to_string(),
        file_size_bytes: file_size,
        block_bytes: block_bytes_report,
        index_file: index_file.map(|path| path.display().to_string()),
        expert_entries_available: available.len(),
        requests: trace.len(),
        seed,
        workload: if trace_file.is_some() { "trace-file".to_string() } else { workload.to_string() },
        trace_source: trace_file.map_or_else(|| format!("synthetic:{}", workload), |path| path.display().to_string()),
        page_cache_controlled: false,
        note: "Real file reads on this host. If --index-file is supplied, each trace key maps through that validated ExpertTable to a variable-length byte range; otherwise fixed-size synthetic layer-0 blocks are used. OS page cache is uncontrolled, estimated backing bytes are summed requested range lengths on cache misses (not physical-device counters), and caller-owned buffers are outside the cache budget. This is not an inference benchmark or a target-hardware claim unless run on the target machine.".to_string(),
        patterns,
    })
}

const MAX_EXPERT_INDEX_BYTES: u64 = 128 * 1024 * 1024;
const MAX_EXPERT_TRACE_BYTES: u64 = 128 * 1024 * 1024;

fn read_bounded_input(
    path: &std::path::Path,
    max_bytes: u64,
    description: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let size = std::fs::metadata(path)?.len();
    if size > max_bytes {
        return Err(format!(
            "{description} file is {size} bytes; input limit is {max_bytes} bytes"
        ).into());
    }
    Ok(std::fs::read(path)?)
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExpertTraceInput {
    FlatIds(Vec<u32>),
    ExpertKeys(Vec<ExpertKey>),
    Pairs(Vec<(u32, u32)>),
}

fn run_prefetch_replay(
    trace_file: &std::path::Path,
    block_bytes: u64,
    top_k: usize,
    max_pending: usize,
    max_inflight_mib: u64,
    latency_requests: usize,
    blob_file: Option<&std::path::Path>,
    index_file: Option<&std::path::Path>,
) -> Result<PrefetchReplayReport, Box<dyn std::error::Error>> {
    use std::collections::HashSet;

    if top_k == 0 || max_pending == 0 || max_inflight_mib == 0 {
        return Err("top-k, max-pending and max-inflight-mib must be greater than zero".into());
    }
    if (blob_file.is_some() != index_file.is_some()) {
        return Err("--blob-file and --index-file must be supplied together".into());
    }
    let indexed_sizes: Option<HashMap<ExpertKey, u64>> = if let (Some(blob_path), Some(index_path)) = (blob_file, index_file) {
        let blob_size = std::fs::metadata(blob_path)?.len();
        let contents = read_bounded_input(index_path, MAX_EXPERT_INDEX_BYTES, "expert index")?;
        let table: ExpertTable = serde_json::from_slice(&contents)?;
        table.validate(blob_size)?;
        if table.ranges.is_empty() {
            return Err("expert index contains no entries".into());
        }
        if table.ranges.len() > 1_000_000 {
            return Err("expert index contains more than one million entries; reduce the index to bound memory".into());
        }
        Some(table.ranges.into_iter().map(|range| (
            ExpertKey { layer: range.layer, expert_id: range.expert_id },
            range.byte_length,
        )).collect())
    } else {
        if block_bytes == 0 {
            return Err("block-bytes must be greater than zero without --index-file".into());
        }
        None
    };

    let input: ExpertTraceInput = serde_json::from_slice(
        &read_bounded_input(trace_file, MAX_EXPERT_TRACE_BYTES, "expert trace")?
    )?;
    let trace: Vec<ExpertKey> = match input {
        ExpertTraceInput::FlatIds(ids) => ids.into_iter()
            .map(|expert_id| ExpertKey { layer: 0, expert_id }).collect(),
        ExpertTraceInput::ExpertKeys(keys) => keys,
        ExpertTraceInput::Pairs(pairs) => pairs.into_iter()
            .map(|(layer, expert_id)| ExpertKey { layer, expert_id }).collect(),
    };
    if trace.is_empty() {
        return Err("trace file must be a non-empty JSON array of expert IDs, [layer, expert_id] pairs, or {layer, expert_id} objects".into());
    }
    if trace.len() > 10_000_000 {
        return Err("trace contains more than 10 million events; split the trace to bound memory".into());
    }
    let distinct: HashSet<ExpertKey> = trace.iter().copied().collect();
    let mut candidates: Vec<(ExpertKey, u64)> = distinct.iter().copied().map(|key| {
        let bytes = indexed_sizes.as_ref()
            .and_then(|sizes| sizes.get(&key).copied())
            .unwrap_or(block_bytes);
        (key, bytes)
    }).collect();
    if let Some(sizes) = &indexed_sizes {
        if let Some(missing) = distinct.iter().find(|key| !sizes.contains_key(key)) {
            return Err(format!("trace references expert ({}, {}), which is absent from the selected index", missing.layer, missing.expert_id).into());
        }
    }
    candidates.sort_by_key(|(key, _)| (key.layer, key.expert_id));
    let distinct_experts_in_trace = candidates.len();
    let candidate_bytes_total = candidates.iter().fold(0u64, |sum, (_, bytes)| sum.saturating_add(*bytes));
    let max_inflight_bytes = max_inflight_mib.checked_mul(1024 * 1024)
        .ok_or("max-inflight-mib is too large")?;
    let mut predictor = ExpertPrefetcher::new(max_pending, max_inflight_bytes);
    let mut scheduled_at: Vec<(ExpertKey, usize)> = Vec::new();
    let mut predicted_candidates = 0u64;
    let mut correct_next_event_predictions = 0u64;

    for (index, actual) in trace.iter().copied().enumerate() {
        let mut remaining = Vec::new();
        for (key, scheduled_index) in scheduled_at.drain(..) {
            if index.saturating_sub(scheduled_index) >= latency_requests {
                predictor.mark_ready(key);
            } else {
                remaining.push((key, scheduled_index));
            }
        }
        scheduled_at = remaining;

        predictor.observe(actual);
        // Discard any stale completion record before this key can be scheduled again.
        scheduled_at.retain(|(key, _)| *key != actual);
        if index + 1 < trace.len() {
            let plans = predictor.predict(actual, &candidates, top_k);
            predicted_candidates = predicted_candidates.saturating_add(plans.len() as u64);
            let next = trace[index + 1];
            if plans.iter().any(|candidate| candidate.key == next) {
                correct_next_event_predictions = correct_next_event_predictions.saturating_add(1);
            }
            for plan in plans {
                scheduled_at.push((plan.key, index));
            }
        }
    }

    predictor.mark_all_unused();
    let stats = predictor.stats();
    let eligible_next_events = trace.len().saturating_sub(1) as f64;
    let next_event_coverage = if eligible_next_events == 0.0 {
        0.0
    } else {
        correct_next_event_predictions as f64 / eligible_next_events
    };
    let next_event_precision = if predicted_candidates == 0 {
        0.0
    } else {
        correct_next_event_predictions as f64 / predicted_candidates as f64
    };
    Ok(PrefetchReplayReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase_gate: "5C-prefetch-trace-replay".to_string(),
        trace_file: trace_file.display().to_string(),
        blob_file: blob_file.map(|path| path.display().to_string()),
        index_file: index_file.map(|path| path.display().to_string()),
        trace_events: trace.len(),
        distinct_experts_in_trace,
        block_bytes: if indexed_sizes.is_some() { None } else { Some(block_bytes) },
        candidate_bytes_total,
        top_k,
        max_pending,
        max_inflight_bytes,
        simulated_latency_requests: latency_requests,
        next_event_coverage,
        next_event_precision,
        predicted_candidates,
        correct_next_event_predictions,
        stats,
        note: if indexed_sizes.is_some() {
            "Trace replay only, using validated variable-size expert ranges from --index-file and --blob-file. Latency is simulated in trace-access steps; no disk reads, live routing capture, or compute/I/O overlap are performed. The candidate universe is limited to experts present in the supplied trace. Do not treat these diagnostics as inference performance.".to_string()
        } else {
            "Trace replay only, using a fixed synthetic block size. Latency is simulated in trace-access steps; no disk reads, live routing capture, or compute/I/O overlap are performed. Trace entries may be flat IDs (layer 0), [layer, expert_id] pairs, or {layer, expert_id} objects; the candidate universe is limited to experts present in the supplied trace. Do not treat these diagnostics as inference performance.".to_string()
        },
    })
}

fn print_prefetch_replay_human(report: &PrefetchReplayReport) {
    println!("KILN prefetch trace replay");
    println!("  trace: {} events from {}", report.trace_events, report.trace_file);
    println!("  next-event coverage: {:.2}% | candidate precision: {:.2}%",
        report.next_event_coverage * 100.0, report.next_event_precision * 100.0);
    println!("  scheduled: {} | useful: {} | late: {} | unused: {} | cancelled: {}",
        report.stats.scheduled, report.stats.useful, report.stats.late,
        report.stats.unused, report.stats.cancelled);
    println!("  max in-flight bytes: {} | current in-flight bytes: {}",
        report.max_inflight_bytes, report.stats.in_flight_bytes);
    if let Some(block_bytes) = report.block_bytes {
        println!("  sizing: fixed {block_bytes}-byte synthetic blocks");
    } else {
        println!("  sizing: indexed ranges from {} (blob {})",
            report.index_file.as_deref().unwrap_or("<unknown>"),
            report.blob_file.as_deref().unwrap_or("<unknown>"));
    }
    println!("  candidate bytes represented: {}", report.candidate_bytes_total);
    println!("  {}", report.note);
}

fn print_expert_cache_human(report: &ExpertCacheReport) {
    println!("KILN expert-cache trace replay");
    println!("  phase gate: {} | file: {} ({} bytes)", report.phase_gate, report.file, report.file_size_bytes);
    if let Some(block_bytes) = report.block_bytes {
        println!("  synthetic block size: {block_bytes} bytes");
    } else {
        println!("  expert index: {}", report.index_file.as_deref().unwrap_or("<unknown>"));
    }
    println!("  entries: {} | requests: {} | workload: {} | seed: {}",
        report.expert_entries_available, report.requests, report.workload, report.seed);
    println!("  OS page cache: uncontrolled");
    for pattern in &report.patterns {
        println!("  {}: {:.2}% hits, {:.2} requests/s, {} estimated backing bytes, {} retained bytes, {} evictions",
            pattern.policy, pattern.hit_rate * 100.0, pattern.requests_per_second,
            pattern.estimated_backing_bytes_read, pattern.cache_used_bytes, pattern.evictions);
    }
    println!("  {}", report.note);
}

fn print_expert_io_human(report: &ExpertIoReport) {
    println!("KILN expert-block I/O feasibility report");
    println!("  phase gate:     {}", report.phase_gate);
    println!("  file:           {}", report.file);
    println!("  file size:      {} bytes", report.file_size_bytes);
    if let Some(index_file) = &report.index_file {
        println!("  expert index:  {} ({} mapped ranges)", index_file, report.indexed_expert_ranges.unwrap_or(0));
    } else {
        println!("  expert index:  none (fixed-offset file probe)");
    }
    println!("  host:           {} / {}", report.os, report.architecture);
    println!("  block size:     {} bytes", report.block_bytes);
    println!("  requests/run:   {}", report.requests_per_run);
    println!("  repetitions:    {}", report.repetitions);
    println!("  random seed:    {}", report.seed);
    println!("  page cache:     uncontrolled (eviction advice only; cannot verify cold reads)");
    for pattern in &report.patterns {
        println!("");
        println!("  {} access", pattern.access_pattern);
        println!("    throughput:   {:.2} MiB/s", pattern.throughput_mib_s);
        println!("    latency p50:  {:.2} us", pattern.request_latency_p50_us);
        println!("    latency p95:  {:.2} us", pattern.request_latency_p95_us);
        println!("    total read:   {} bytes", pattern.total_bytes_read);
        println!("    cache hint:   {}", if pattern.page_cache_eviction_advised { "POSIX_FADV_DONTNEED issued before pattern" } else { "none" });
        println!("    request size: {}..{} bytes", pattern.min_request_bytes, pattern.max_request_bytes);
        if !pattern.expert_keys.is_empty() {
            println!("    mapped expert requests: {}", pattern.expert_keys.len());
        }
    if let Some(bytes) = pattern.linux_process_storage_read_bytes {
        println!("    Linux process storage read: {} bytes (page-cache influenced)", bytes);
    }
    }
    println!("");
    println!("  {}", report.note);
}

fn measure<F: FnOnce() -> f64>(name: &str, unit: &str, f: F) -> Measurement {
    let start = Instant::now();
    let value = f();
    let elapsed = start.elapsed().as_secs_f64();
    Measurement {
        name: name.to_string(),
        value,
        unit: unit.to_string(),
        success: true,
        notes: format!("measured in {:.6} s", elapsed),
    }
}

fn run_benchmarks() -> BenchReport {
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut measurements = Vec::new();

    // Bench 1: monitor construction time.
    let mut monitor_holder: Option<LinuxMonitor> = None;
    measurements.push(measure("monitor_new", "nanos", || {
        let start = Instant::now();
        let m = LinuxMonitor::new();
        monitor_holder = Some(m);
        start.elapsed().as_nanos() as f64
    }));

    // Bench 2: monitor sample time.
    measurements.push(measure("monitor_sample", "nanos", || {
        if let Some(ref mut m) = monitor_holder {
            let start = Instant::now();
            m.sample();
            start.elapsed().as_nanos() as f64
        } else {
            0.0
        }
    }));

    // Bench 3: envelope read time.
    measurements.push(measure("monitor_envelope_read", "nanos", || {
        if let Some(ref m) = monitor_holder {
            let start = Instant::now();
            let _env = m.current_envelope();
            start.elapsed().as_nanos() as f64
        } else {
            0.0
        }
    }));

    // Bench 4: thermal history read time.
    measurements.push(measure("monitor_history_read", "nanos", || {
        if let Some(ref m) = monitor_holder {
            let start = Instant::now();
            let _h = m.thermal_history(std::time::Duration::from_secs(60));
            start.elapsed().as_nanos() as f64
        } else {
            0.0
        }
    }));

    // Bench 5: throttle prediction time.
    measurements.push(measure("monitor_predict", "nanos", || {
        if let Some(ref m) = monitor_holder {
            let start = Instant::now();
            let _p = m.predicted_throttle();
            start.elapsed().as_nanos() as f64
        } else {
            0.0
        }
    }));

    // Bench 6: TQ1.0 pack throughput.
    let n = 1_000_000usize;
    let test_vals: Vec<f32> = (0..n)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();

    let _ = kiln_kernels::pack(&test_vals);

    let iters = 10usize;
    measurements.push(measure("pack_tq1_0_ns_per_element", "nanos/elem", || {
        let start = Instant::now();
        for _ in 0..iters {
            let packed = kiln_kernels::pack(&test_vals);
            std::hint::black_box(packed);
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        elapsed / (iters as f64 * n as f64)
    }));

    let packed = kiln_kernels::pack(&test_vals);
    measurements.push(measure("unpack_tq1_0_ns_per_element", "nanos/elem", || {
        let start = Instant::now();
        for _ in 0..iters {
            let unpacked = kiln_kernels::unpack(&packed, n);
            std::hint::black_box(unpacked);
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        elapsed / (iters as f64 * n as f64)
    }));

    let bytes_in = (n * 4) as f64;
    let bytes_out = (n / 5) as f64;
    measurements.push(measure("pack_tq1_0_mb_per_sec_input", "MB/s", || {
        let start = Instant::now();
        for _ in 0..iters {
            let p = kiln_kernels::pack(&test_vals);
            std::hint::black_box(p);
        }
        let elapsed = start.elapsed().as_secs_f64();
        (bytes_in * iters as f64) / elapsed / 1_000_000.0
    }));

    measurements.push(measure("pack_tq1_0_mb_per_sec_output", "MB/s", || {
        let start = Instant::now();
        for _ in 0..iters {
            let p = kiln_kernels::pack(&test_vals);
            std::hint::black_box(p);
        }
        let elapsed = start.elapsed().as_secs_f64();
        (bytes_out * iters as f64) / elapsed / 1_000_000.0
    }));

    // Bench 8b: table-based unpack throughput.
    measurements.push(measure("unpack_tq1_0_table_ns_per_element", "nanos/elem", || {
        let start = Instant::now();
        for _ in 0..iters {
            let u = kiln_kernels::unpack_table(&packed, n);
            std::hint::black_box(u);
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        elapsed / (iters as f64 * n as f64)
    }));

    // Bench 8c: unpack table speedup ratio.
    let unpack_scalar_ns = {
        let start = Instant::now();
        for _ in 0..iters {
            let u = kiln_kernels::unpack(&packed, n);
            std::hint::black_box(u);
        }
        start.elapsed().as_nanos() as f64 / (iters as f64 * n as f64)
    };
    let unpack_table_ns = {
        let start = Instant::now();
        for _ in 0..iters {
            let u = kiln_kernels::unpack_table(&packed, n);
            std::hint::black_box(u);
        }
        start.elapsed().as_nanos() as f64 / (iters as f64 * n as f64)
    };
    measurements.push(Measurement {
        name: "unpack_tq1_0_table_speedup".to_string(),
        value: unpack_scalar_ns / unpack_table_ns,
        unit: "x".to_string(),
        success: unpack_table_ns < unpack_scalar_ns,
        notes: format!("scalar {:.2} ns/elem, table {:.2} ns/elem", unpack_scalar_ns, unpack_table_ns),
    });

    // Bench 9: pack scalar path, for speedup comparison.
    let scalar_ns = {
        let start = Instant::now();
        for _ in 0..iters {
            let p = kiln_kernels::pack_scalar(&test_vals);
            std::hint::black_box(p);
        }
        start.elapsed().as_nanos() as f64 / (iters as f64 * n as f64)
    };
    measurements.push(Measurement {
        name: "pack_tq1_0_scalar_ns_per_element".to_string(),
        value: scalar_ns,
        unit: "nanos/elem".to_string(),
        success: scalar_ns > 0.0,
        notes: format!("scalar path only, {} iterations", iters),
    });

    // Bench 10: pack AVX2 path, measured separately.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let avx2_ns = {
            let start = Instant::now();
            for _ in 0..iters {
                let p = kiln_kernels::pack_avx2(&test_vals);
                std::hint::black_box(p);
            }
            start.elapsed().as_nanos() as f64 / (iters as f64 * n as f64)
        };
        measurements.push(Measurement {
            name: "pack_tq1_0_avx2_ns_per_element".to_string(),
            value: avx2_ns,
            unit: "nanos/elem".to_string(),
            success: avx2_ns > 0.0,
            notes: format!("AVX2 path only, {} iterations", iters),
        });

        // Bench 12: fused ternary matmul scalar throughput.
    // Weight row length 1024, 1000 iterations, measured per output.
    let matmul_n = 1024usize;
    let matmul_iters = 1000usize;
    let w_vals: Vec<f32> = (0..matmul_n)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    let w_packed = kiln_kernels::pack(&w_vals);
    let acts: Vec<i8> = (0..matmul_n)
        .map(|i| ((i as i32 * 7 % 200) - 100) as i8)
        .collect();

    measurements.push(measure("matmul_fused_scalar_ns_per_call", "nanos/call", || {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..matmul_iters {
            sum += kiln_kernels::matmul_scalar(&w_packed, &acts, matmul_n, 1.0);
        }
        std::hint::black_box(sum);
        let elapsed = start.elapsed().as_nanos() as f64;
        elapsed / matmul_iters as f64
    }));

    // Bench 13: fused ternary matmul AVX2 throughput.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    measurements.push(measure("matmul_fused_lut_naive_ns_per_call", "nanos/call", || {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..matmul_iters {
            sum += kiln_kernels::matmul_lut_naive(&w_packed, &acts, matmul_n, 1.0);
        }
        std::hint::black_box(sum);
        let elapsed = start.elapsed().as_nanos() as f64;
        elapsed / matmul_iters as f64
    }));

    // Bench 14: fused matmul speedup ratio.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let scalar_ns = {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..matmul_iters {
                sum += kiln_kernels::matmul_scalar(&w_packed, &acts, matmul_n, 1.0);
            }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / matmul_iters as f64
        };
        let avx2_ns = {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..matmul_iters {
                sum += kiln_kernels::matmul_lut_naive(&w_packed, &acts, matmul_n, 1.0);
            }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / matmul_iters as f64
        };
        measurements.push(Measurement {
            name: "matmul_fused_lut_naive_speedup".to_string(),
            value: scalar_ns / avx2_ns,
            unit: "x".to_string(),
            success: avx2_ns < scalar_ns,
            notes: format!("scalar {:.0} ns/call, avx2 {:.0} ns/call at n={}", scalar_ns, avx2_ns, matmul_n),
        });
    }

    // Bench 12b: fused matmul 8-row AVX2 throughput.
    // 8 weight rows of length 1024, shared activation vector.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let stride = (matmul_n + 4) / 5;
        let mut weights_8x = vec![0u8; stride * 8];
        for r in 0..8 {
            let w: Vec<f32> = (0..matmul_n)
                .map(|i| match (i + r * 3) % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
                .collect();
            let p = kiln_kernels::pack(&w);
            for b in 0..stride {
                weights_8x[r * stride + b] = p[b];
            }
        }

        measurements.push(measure("matmul_fused_8row_lut_failed_ns_per_call", "nanos/call", || {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..matmul_iters {
                let out = kiln_kernels::matmul_8row_lut_failed(&weights_8x, stride, matmul_n, &acts, 1.0);
                for v in out.iter() { sum += v; }
            }
            std::hint::black_box(sum);
            let elapsed = start.elapsed().as_nanos() as f64;
            elapsed / matmul_iters as f64
        }));

        // Speedup vs scalar for the same 8-row workload.
        let scalar_8row_ns = {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..matmul_iters {
                for r in 0..8 {
                    let row_start = r * stride;
                    let row_end = row_start + stride;
                    sum += kiln_kernels::matmul_scalar(
                        &weights_8x[row_start..row_end],
                        &acts,
                        matmul_n,
                        1.0,
                    );
                }
            }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / matmul_iters as f64
        };

        let avx2_8row_ns = {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..matmul_iters {
                let out = kiln_kernels::matmul_8row_lut_failed(&weights_8x, stride, matmul_n, &acts, 1.0);
                for v in out.iter() { sum += v; }
            }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / matmul_iters as f64
        };

        measurements.push(Measurement {
            name: "matmul_fused_8row_lut_failed_speedup".to_string(),
            value: scalar_8row_ns / avx2_8row_ns,
            unit: "x".to_string(),
            success: avx2_8row_ns < scalar_8row_ns,
            notes: format!(
                "8-row scalar {:.0} ns/call, 8-row AVX2 {:.0} ns/call at n={}",
                scalar_8row_ns, avx2_8row_ns, matmul_n
            ),
        });
    }

    // Bench 11: speedup ratio.
        let ratio = scalar_ns / avx2_ns;
        measurements.push(Measurement {
            name: "pack_tq1_0_avx2_speedup".to_string(),
            value: ratio,
            unit: "x".to_string(),
            success: ratio > 1.0,
            notes: format!("scalar {:.2} ns/elem / avx2 {:.2} ns/elem", scalar_ns, avx2_ns),
        });
    }

    let passed = measurements.iter().filter(|m| m.success).count() as u32;
    let failed = measurements.iter().filter(|m| !m.success).count() as u32;

    BenchReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase: "0B".to_string(),
        started_at_unix: started,
        measurements,
        summary: Summary {
            total: passed + failed,
            passed,
            failed,
        },
    }
}

fn print_human(report: &BenchReport) {
    println!("KILN benchmark report");
    println!("  version:    {}", report.version);
    println!("  phase:      {}", report.phase);
    println!("  started:    {}", report.started_at_unix);
    println!("");
    println!("  {:<28} {:>14} {:<8} {}", "name", "value", "unit", "notes");
    println!("  {}", "-".repeat(70));
    for m in &report.measurements {
        println!(
            "  {:<28} {:>14.2} {:<8} {}",
            m.name, m.value, m.unit, m.notes
        );
    }
    println!("");
    println!(
        "  summary: {}/{} passed, {} failed",
        report.summary.passed, report.summary.total, report.summary.failed
    );
}

fn print_info() {
    println!("KILN benchmark harness");
    println!("  Phase 0B measures:");
    println!("    - monitor_new:        construction time for LinuxMonitor");
    println!("    - monitor_sample:     time to read /proc and /sys once");
    println!("    - monitor_envelope_read: time to read current envelope");
    println!("    - monitor_history_read:  time to build a thermal history window");
    println!("    - monitor_predict:    time to run throttle prediction");
    println!("");
    println!("");
    println!("  Phase 1 measures (kernels):");
    println!("    - pack_tq1_0_ns_per_element:   pack 1M trits, ns per element");
    println!("    - unpack_tq1_0_ns_per_element: unpack 1M trits, ns per element");
    println!("    - pack_tq1_0_mb_per_sec_input:  pack throughput, MB/s input");
    println!("    - pack_tq1_0_mb_per_sec_output: pack throughput, MB/s output");
    println!("    - pack_tq1_0_scalar_ns_per_element: scalar only, ns per element");
    println!("    - pack_tq1_0_avx2_ns_per_element:   AVX2 only, ns per element");
    println!("    - pack_tq1_0_avx2_speedup:          ratio, scalar / AVX2");
    println!("");
    println!("  Phase 1 will add:");
    println!("    - decode_tokens_per_sec: for each model in the catalog");
    println!("    - prefill_tokens_per_sec: for each model in the catalog");
    println!("    - time_to_first_token_ms: for each model in the catalog");
    println!("    - peak_ram_mb:           per model");
    println!("    - wikitext2_perplexity:  per model");
    println!("    - mmlu_score:            per model");
    println!("");
    println!("  Phase 5A: expert-io --file <path> measures bounded sequential/random file reads");
    println!("  Phase 5B: expert-cache --file <path> replays synthetic expert traces across cache policies");
    println!("  Both are storage/cache probes, not end-to-end model throughput benchmarks.");
    println!("");
    println!("  Phase 2 through 6 extend the harness with the acceptance tests");
    println!("  AT-1 through AT-8 defined in Section 13 of the Master File.");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { json } => {
            let report = run_benchmarks();
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                print_human(&report);
            }
        }
        Commands::Info => {
            print_info();
        }
        Commands::ExpertCache { file, block_bytes, requests, cache_mib, seed, workload, index_file, trace_file, json } => {
            match run_expert_cache(&file, block_bytes, requests, cache_mib, seed, &workload, trace_file.as_deref(), index_file.as_deref()) {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => print_expert_cache_human(&report),
                Err(error) => {
                    eprintln!("expert cache benchmark failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::PrefetchReplay { trace_file, blob_file, index_file, block_bytes, top_k, max_pending, max_inflight_mib, latency_requests, json } => {
            match run_prefetch_replay(&trace_file, block_bytes, top_k, max_pending, max_inflight_mib, latency_requests, blob_file.as_deref(), index_file.as_deref()) {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => print_prefetch_replay_human(&report),
                Err(error) => {
                    eprintln!("prefetch replay failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::PredictorHoldout { trace_file, train_events, top_k, json } => {
            let report = (|| -> Result<_, Box<dyn std::error::Error>> {
                let input: ExpertTraceInput = serde_json::from_slice(
                    &read_bounded_input(&trace_file, MAX_EXPERT_TRACE_BYTES, "expert trace")?
                )?;
                let trace: Vec<ExpertKey> = match input {
                    ExpertTraceInput::FlatIds(ids) => ids.into_iter()
                        .map(|expert_id| ExpertKey { layer: 0, expert_id }).collect(),
                    ExpertTraceInput::ExpertKeys(keys) => keys,
                    ExpertTraceInput::Pairs(pairs) => pairs.into_iter()
                        .map(|(layer, expert_id)| ExpertKey { layer, expert_id }).collect(),
                };
                if trace.len() > 10_000_000 {
                    return Err("expert trace too large; split before evaluation".into());
                }
                Ok(evaluate_predictor_holdout(&trace, train_events, top_k)?)
            })();
            match report {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => {
                    println!("KILN Phase 5.0 chronological predictor holdout");
                    println!("  Train / heldout: {} / {}", report.train_events, report.test_events);
                    println!("  Accuracy@{}: {:.2}%", report.top_k, 100.0*report.accuracy_at_k);
                    println!("  Coverage: {:.2}%", 100.0*report.coverage);
                    println!("  Train-only unigram baseline: {:.2}%", 100.0*report.unigram_baseline_accuracy_at_k);
                    println!("  Reaches exploratory 80% criterion: {}", report.passed_80_percent_feasibility_threshold);
                    println!("  Only real-model traces can establish target feasibility.");
                }
                Err(error) => {
                    eprintln!("predictor holdout evaluation failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::RouterHoldout { trace_file, train_until_token, top_k, json } => {
            let report = (|| -> Result<_, Box<dyn std::error::Error>> {
                let raw=read_bounded_input(&trace_file, MAX_EXPERT_TRACE_BYTES, "router trace")?;
                let events:Vec<RouterEvent>=serde_json::from_slice(&raw)?;
                if events.len()>10_000_000 { return Err("router trace exceeds event limit".into()); }
                Ok(evaluate_grouped_router_holdout(&events,train_until_token,top_k)?)
            })();
            match report {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => {
                    println!("KILN Phase 5.0 grouped router holdout");
                    println!("  Events train/test: {}/{}",report.train_events,report.test_events);
                    println!("  Complete active-expert set accuracy@{}: {:.2}%",top_k,report.accuracy_at_k*100.0);
                    println!("  Event coverage: {:.2}%",report.coverage*100.0);
                    println!("  Individual active-expert recall: {:.2}%",report.active_expert_recall_at_k*100.0);
                    println!("  Train-only unigram baseline: {:.2}%",report.unigram_baseline_accuracy_at_k*100.0);
                    println!("  Real-model + Tier 0 validation remains required.");
                }
                Err(e) => {
                    eprintln!("grouped router holdout failed: {e}");
                    std::process::exit(2);
                }
            }
        }
        Commands::ExpertPack { source_file, source_index, blob_file,
            index_file, integrity_file, json } => {
            let result = (|| -> Result<_, Box<dyn std::error::Error>> {
                let index_bytes = read_bounded_input(
                    &source_index, MAX_EXPERT_INDEX_BYTES, "source expert index"
                )?;
                let source_table: ExpertTable = serde_json::from_slice(&index_bytes)?;
                let manifest = write_expert_sidecar(
                    &source_file, source_table, &blob_file, &index_file, &integrity_file
                )?;
                // Verify all bytes before reporting a completed pack.
                let table = verify_expert_sidecar(&blob_file, &index_file, &integrity_file)?;
                Ok((manifest, table))
            })();
            match result {
                Ok((manifest, table)) if json => {
                    println!("{}", serde_json::json!({
                        "status":"packed_and_crc32_verified",
                        "expert_count":table.ranges.len(),
                        "packed_blob_bytes":table.blob_size,
                        "source_size_bytes":manifest.source_size_bytes,
                        "alignment_bytes":manifest.alignment_bytes
                    }));
                }
                Ok((manifest,table)) => {
                    println!("KILN Phase 5.1 sidecar packed and verified");
                    println!("  Experts: {}", table.ranges.len());
                    println!("  Packed bytes: {}", table.blob_size);
                    println!("  Alignment: {}", manifest.alignment_bytes);
                    println!("  Original GGUF/source file unchanged");
                }
                Err(error) => {
                    eprintln!("sidecar pack failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::ExpertPackNamed { source_gguf, selections_file, blob_file,
            index_file, integrity_file, components_file, json } => {
            let result = (|| -> Result<_, Box<dyn std::error::Error>> {
                let choices: Vec<NamedExpertTensor> = serde_json::from_slice(
                    &read_bounded_input(&selections_file, MAX_EXPERT_INDEX_BYTES, "GGUF named tensor selections")?
                )?;
                let map = resolve_expert_tensors_in_gguf(&source_gguf, &choices)?;
                let (integrity, components) = write_component_expert_sidecar(
                    &source_gguf, map, &blob_file, &index_file, &integrity_file, &components_file
                )?;
                let (table, verified) = verify_component_expert_sidecar(
                    &blob_file, &index_file, &integrity_file, &components_file
                )?;
                Ok((table, integrity, components, verified))
            })();
            match result {
                Ok((table, integrity, components, verified)) if json => {
                    println!("{}", serde_json::json!({
                        "status":"named_gguf_expert_components_crc32_verified",
                        "experts":table.ranges.len(),
                        "components":verified.components.len(),
                        "source_bytes":integrity.source_size_bytes,
                        "sidecar_bytes":table.blob_size,
                        "metadata_matches":components.components.len()==verified.components.len()
                    }));
                }
                Ok((table, _integrity, _components, verified)) => {
                    println!("KILN Phase 5.1 named GGUF expert components packed and verified");
                    println!("  Expert groups: {}",table.ranges.len());
                    println!("  Tensor components: {}",verified.components.len());
                    println!("  Sidecar bytes: {}",table.blob_size);
                    println!("  Original GGUF unchanged");
                }
                Err(e) => {
                    eprintln!("named GGUF expert pack failed: {e}");
                    std::process::exit(2);
                }
            }
        }
        Commands::ExpertPackVerify { blob_file, index_file, integrity_file, json } => {
            match verify_expert_sidecar(&blob_file, &index_file, &integrity_file) {
                Ok(table) if json => println!("{}", serde_json::json!({
                    "status":"crc32_verified",
                    "expert_count":table.ranges.len(),
                    "blob_size":table.blob_size
                })),
                Ok(table) => println!(
                    "KILN sidecar verified: {} experts, {} bytes",
                    table.ranges.len(), table.blob_size
                ),
                Err(error) => {
                    eprintln!("sidecar integrity check failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::ExpertIo { file, index_file, trace_file, block_bytes, requests, runs, seed, advise_drop_cache, json } => {
            let result = match (index_file.as_deref(), trace_file.as_deref()) {
                (Some(index), Some(trace)) => run_expert_io_trace_with_advice(&file, index, trace, block_bytes, runs, advise_drop_cache),
                (Some(index), None) => run_expert_io_indexed_with_advice(&file, index, block_bytes, requests, runs, seed, advise_drop_cache),
                (None, Some(_)) => Err("--trace-file requires --index-file".into()),
                (None, None) if advise_drop_cache => Err("--advise-drop-cache requires a validated --index-file".into()),
                (None, None) => run_expert_io(&file, block_bytes, requests, runs, seed),
            };
            match result {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => print_expert_io_human(&report),
                Err(error) => {
                    eprintln!("expert I/O benchmark failed: {error}");
                    std::process::exit(2);
                }
            }
        }
    }
}


#[cfg(test)]
mod expert_io_tests {
    use super::*;
    use std::io::Write;

    fn fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "kiln-expert-io-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        let bytes: Vec<u8> = (0..=255).cycle().take(64 * 1024).collect();
        file.write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn expert_io_report_measures_sequential_and_random_reads() {
        let path = fixture();
        let report = run_expert_io(&path, 1024, 8, 2, 42).unwrap();
        assert_eq!(report.patterns.len(), 2);
        assert_eq!(report.patterns[0].access_pattern, "sequential");
        assert_eq!(report.patterns[1].access_pattern, "random");
        assert_eq!(report.patterns[0].total_bytes_read, 16 * 1024);
        assert_eq!(report.patterns[0].min_request_bytes, 1024);
        assert_eq!(report.patterns[0].max_request_bytes, 1024);
        assert!(report.index_file.is_none());
        assert!(report.patterns.iter().all(|p| p.throughput_mib_s > 0.0));
        std::fs::remove_file(path).unwrap();
    }

    fn write_expert_index(path: &std::path::Path, blob_size: u64) -> PathBuf {
        let index_path = path.with_extension("experts.index.json");
        let ranges = (0..8).map(|expert_id| ExpertRange {
            layer: expert_id / 4,
            expert_id: expert_id % 4,
            byte_offset: u64::from(expert_id) * 8192,
            byte_length: 4096,
        }).collect();
        let table = ExpertTable { blob_version: 1, blob_size, ranges };
        std::fs::write(&index_path, serde_json::to_vec(&table).unwrap()).unwrap();
        index_path
    }

    #[test]
    fn indexed_expert_io_reads_only_validated_model_expert_ranges() {
        let path = fixture();
        let index_path = write_expert_index(&path, 64 * 1024);
        let report = run_expert_io_indexed(&path, &index_path, 1024, 4, 2, 42).unwrap();
        assert_eq!(report.indexed_expert_ranges, Some(8));
        let index_path_string = index_path.display().to_string();
        assert_eq!(report.index_file.as_deref(), Some(index_path_string.as_str()));
        assert_eq!(report.patterns[0].access_pattern, "sequential");
        assert_eq!(report.patterns[1].access_pattern, "random");
        assert_eq!(report.patterns[0].total_bytes_read, 8 * 1024);
        assert_eq!(report.patterns[0].expert_keys.len(), 4);
        assert!(report.patterns.iter().all(|pattern| pattern.expert_keys.iter().all(|key| key.layer < 2)));
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn indexed_expert_io_rejects_index_for_different_blob_size() {
        let path = fixture();
        let index_path = write_expert_index(&path, 64 * 1024 - 1);
        assert!(run_expert_io_indexed(&path, &index_path, 1024, 4, 1, 42).is_err());
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn indexed_expert_io_clips_reads_to_each_experts_real_range() {
        let path = fixture();
        let index_path = path.with_extension("variable.experts.index.json");
        let table = ExpertTable {
            blob_version: 1,
            blob_size: 64 * 1024,
            ranges: vec![
                ExpertRange { layer: 0, expert_id: 0, byte_offset: 0, byte_length: 512 },
                ExpertRange { layer: 0, expert_id: 1, byte_offset: 8192, byte_length: 4096 },
            ],
        };
        std::fs::write(&index_path, serde_json::to_vec(&table).unwrap()).unwrap();

        let report = run_expert_io_indexed(&path, &index_path, 1024, 2, 1, 42).unwrap();
        let sequential = &report.patterns[0];
        assert_eq!(sequential.min_request_bytes, 512);
        assert_eq!(sequential.max_request_bytes, 1024);
        assert_eq!(sequential.total_bytes_read, 512 + 1024);
        assert_eq!(sequential.expert_keys.len(), 2);
        let random = &report.patterns[1];
        assert!(random.min_request_bytes == 512 || random.min_request_bytes == 1024);
        assert!(random.max_request_bytes == 512 || random.max_request_bytes == 1024);
        assert!(random.total_bytes_read == 2 * 512 || random.total_bytes_read == 512 + 1024 || random.total_bytes_read == 2 * 1024);
        assert_eq!(random.expert_keys.len(), 2);

        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_io_trace_replays_real_order_and_rejects_missing_experts() {
        let path = fixture();
        let index_path = write_expert_index(&path, 64 * 1024);
        let trace_path = path.with_extension("router-trace.json");
        std::fs::write(&trace_path,
            r#"[{"layer":0,"expert_id":1},{"layer":1,"expert_id":0},{"layer":0,"expert_id":1}]"#).unwrap();
        let report = run_expert_io_trace(&path, &index_path, &trace_path, 4096, 2).unwrap();
        assert_eq!(report.patterns.len(), 1);
        let p = &report.patterns[0];
        assert_eq!(p.access_pattern, "router-trace");
        assert_eq!(p.requests, 6);
        assert_eq!(p.total_bytes_read, 6 * 4096);
        assert_eq!(p.expert_keys, vec![
            ExpertKey { layer: 0, expert_id: 1 },
            ExpertKey { layer: 1, expert_id: 0 },
            ExpertKey { layer: 0, expert_id: 1 },
        ]);
        std::fs::write(&trace_path, r#"[[2,0]]"#).unwrap();
        assert!(run_expert_io_trace(&path, &index_path, &trace_path, 1024, 1).is_err());
        std::fs::write(&trace_path, "[]").unwrap();
        assert!(run_expert_io_trace(&path, &index_path, &trace_path, 1024, 1).is_err());
        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn indexed_file_cache_advice_preserves_expert_bytes_and_measurements() {
        let path = fixture();
        let index_path = write_expert_index(&path, 64 * 1024);
        #[cfg(target_os = "linux")]
        {
            let report = run_expert_io_indexed_with_advice(&path, &index_path, 4096, 4, 1, 42, true).unwrap();
            assert!(report.patterns.iter().all(|pattern| pattern.page_cache_eviction_advised));
            assert!(report.patterns.iter().all(|pattern| pattern.requests == 4));
            assert!(!report.page_cache_controlled);
        }
        #[cfg(not(target_os = "linux"))]
        assert!(run_expert_io_indexed_with_advice(&path, &index_path, 4096, 4, 1, 42, true).is_err());
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_io_rejects_a_block_larger_than_the_file() {
        let path = fixture();
        assert!(run_expert_io(&path, 128 * 1024, 1, 1, 42).is_err());
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod expert_cache_bench_tests {
    use super::*;
    use std::io::Write;

    fn fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "kiln-expert-cache-bench-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes: Vec<u8> = (0..=255).cycle().take(64 * 1024).collect();
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn expert_cache_replay_compares_policies_on_same_trace_and_budget() {
        let path = fixture();
        let report = run_expert_cache(&path, 1024, 500, 1, 42, "stationary-hot", None, None).unwrap();
        assert_eq!(report.patterns.len(), 4);
        assert_eq!(report.patterns[0].policy, "none");
        assert_eq!(report.patterns[0].misses, 500);
        assert_eq!(report.patterns[0].hits, 0);
        assert!(report.patterns[1..].iter().all(|p| p.cache_capacity_bytes == 1024 * 1024));
        assert!(report.patterns[1..].iter().all(|p| p.cache_used_bytes <= p.cache_capacity_bytes));
        assert!(report.patterns[1..].iter().any(|p| p.hits > 0));
        assert!(report.patterns.iter().all(|p| p.checksum == report.patterns[0].checksum));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_cache_replays_supplied_json_trace_and_rejects_unknown_ids() {
        let path = fixture();
        let trace_path = std::env::temp_dir().join(format!(
            "kiln-expert-trace-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&trace_path, "[0,0,1,1,0]").unwrap();
        let report = run_expert_cache(
            &path, 1024, 0, 1, 42, "uniform", Some(&trace_path), None
        ).unwrap();
        assert_eq!(report.requests, 5);
        assert_eq!(report.workload, "trace-file");
        assert_eq!(report.trace_source, trace_path.display().to_string());
        assert!(report.patterns.iter().all(|p| p.checksum == report.patterns[0].checksum));

        std::fs::write(&trace_path, "[64]").unwrap();
        assert!(run_expert_cache(
            &path, 1024, 10, 1, 42, "uniform", Some(&trace_path), None
        ).is_err());
        std::fs::write(&trace_path, "[]").unwrap();
        assert!(run_expert_cache(
            &path, 1024, 10, 1, 42, "uniform", Some(&trace_path), None
        ).is_err());

        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_cache_uses_validated_index_and_layer_aware_trace() {
        let path = fixture();
        let index_path = std::env::temp_dir().join(format!(
            "kiln-expert-index-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let trace_path = std::env::temp_dir().join(format!(
            "kiln-indexed-trace-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let table = ExpertTable {
            blob_version: 1,
            blob_size: std::fs::metadata(&path).unwrap().len(),
            ranges: vec![
                ExpertRange { layer: 0, expert_id: 0, byte_offset: 0, byte_length: 1024 },
                ExpertRange { layer: 0, expert_id: 1, byte_offset: 1024, byte_length: 2048 },
                ExpertRange { layer: 1, expert_id: 0, byte_offset: 3072, byte_length: 512 },
            ],
        };
        std::fs::write(&index_path, serde_json::to_vec(&table).unwrap()).unwrap();
        std::fs::write(
            &trace_path,
            "[{\"layer\":0,\"expert_id\":0},{\"layer\":1,\"expert_id\":0},{\"layer\":0,\"expert_id\":0}]",
        ).unwrap();

        let report = run_expert_cache(
            &path, 0, 0, 1, 42, "uniform", Some(&trace_path), Some(&index_path)
        ).unwrap();
        assert_eq!(report.block_bytes, None);
        assert_eq!(report.index_file, Some(index_path.display().to_string()));
        assert_eq!(report.expert_entries_available, 3);
        assert_eq!(report.requests, 3);
        assert_eq!(report.patterns[0].misses, 3);
        assert_eq!(report.patterns[0].estimated_backing_bytes_read, 2560);
        assert_eq!(report.patterns[1].hits, 1);
        assert_eq!(report.patterns[1].misses, 2);
        assert!(report.patterns.iter().all(|p| p.checksum == report.patterns[0].checksum));

        std::fs::write(&trace_path, "[{\"layer\":9,\"expert_id\":99}]").unwrap();
        assert!(run_expert_cache(
            &path, 0, 1, 1, 42, "uniform", Some(&trace_path), Some(&index_path)
        ).is_err());
        std::fs::remove_file(trace_path).unwrap();
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prefetch_replay_reports_predictions_and_rejects_invalid_traces() {
        let trace_path = std::env::temp_dir().join(format!(
            "kiln-prefetch-trace-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&trace_path, "[0,1,0,1,0,1,0,1]").unwrap();
        let report = run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0, None, None).unwrap();
        assert_eq!(report.trace_events, 8);
        assert_eq!(report.distinct_experts_in_trace, 2);
        std::fs::write(&trace_path, "[[0,1],[1,1],[0,1],[1,2],[0,1]]").unwrap();
        let layered = run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0, None, None).unwrap();
        assert_eq!(layered.trace_events, 5);
        assert_eq!(layered.distinct_experts_in_trace, 3);
        assert!(report.next_event_coverage > 0.0);
        assert!(report.predicted_candidates > 0);
        assert_eq!(report.stats.scheduled, report.stats.useful + report.stats.late + report.stats.unused + report.stats.cancelled);
        std::fs::write(&trace_path, "[]").unwrap();
        assert!(run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0, None, None).is_err());
        std::fs::remove_file(trace_path).unwrap();
    }

    #[test]
    fn expert_cache_replay_rejects_invalid_inputs() {
        let path = fixture();
        assert!(run_expert_cache(&path, 0, 10, 1, 42, "uniform", None, None).is_err());
        assert!(run_expert_cache(&path, 1024, 10, 0, 42, "uniform", None, None).is_err());
        assert!(run_expert_cache(&path, 1024, 10, 1, 42, "unknown", None, None).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
