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

use std::time::Instant;
use std::path::PathBuf;

use kiln_io::{CachedExpertBlobReader, FileRangeReader};
use kiln_mem::{ExpertCachePolicy, ExpertKey, ExpertPrefetcher, PrefetchStats};
use kiln_models::{ExpertRange, ExpertTable};

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
        /// JSON array of expert IDs in observed access order.
        #[arg(long)]
        trace_file: PathBuf,
        /// Fixed expert block size used for planner budget accounting.
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
    /// Measure bounded random and sequential reads from a real model/expert blob.
    ExpertIo {
        /// Path to an expert blob or other representative large model file.
        #[arg(long)]
        file: PathBuf,
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
    total_bytes_read: u64,
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
    trace_events: usize,
    distinct_experts_in_trace: usize,
    block_bytes: u64,
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

fn run_expert_io_pattern(
    reader: &mut FileRangeReader,
    offsets: &[u64],
    block_bytes: usize,
    repetitions: usize,
    access_pattern: &str,
) -> Result<ExpertIoPattern, Box<dyn std::error::Error>> {
    let mut buffer = vec![0u8; block_bytes];
    let mut latencies = Vec::with_capacity(offsets.len().saturating_mul(repetitions));
    let mut checksum = 0u64;
    let total_start = Instant::now();

    for _ in 0..repetitions {
        for offset in offsets {
            let start = Instant::now();
            reader.read_exact_at(*offset, &mut buffer)?;
            latencies.push(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            // Touch a deterministic sample of the bytes to prevent this from
            // becoming a pure seek-only benchmark.
            for byte in buffer.iter().step_by(4096) {
                checksum = checksum.rotate_left(5) ^ u64::from(*byte);
            }
        }
    }

    let elapsed = total_start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let total_bytes_read = (block_bytes as u64)
        .saturating_mul(offsets.len() as u64)
        .saturating_mul(repetitions as u64);
    latencies.sort_unstable();

    Ok(ExpertIoPattern {
        access_pattern: access_pattern.to_string(),
        requests: offsets.len().saturating_mul(repetitions),
        block_bytes,
        total_bytes_read,
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
    if block_bytes == 0 || requests == 0 || runs == 0 {
        return Err("block-bytes, requests, and runs must all be greater than zero".into());
    }
    let mut reader = FileRangeReader::open(file)?;
    let blocks_available = reader.file_size() / block_bytes as u64;
    if blocks_available == 0 {
        return Err(format!(
            "file is {} bytes; it must contain at least one full {}-byte block",
            reader.file_size(), block_bytes
        ).into());
    }

    let mut sequential_offsets = Vec::with_capacity(requests);
    let mut random_offsets = Vec::with_capacity(requests);
    let mut state = seed;
    for i in 0..requests {
        sequential_offsets.push((i as u64 % blocks_available) * block_bytes as u64);
        // Stable LCG: repeatable across platforms and Rust versions.
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        random_offsets.push((state % blocks_available) * block_bytes as u64);
    }

    let sequential = run_expert_io_pattern(&mut reader, &sequential_offsets, block_bytes, runs, "sequential")?;
    let random = run_expert_io_pattern(&mut reader, &random_offsets, block_bytes, runs, "random")?;

    Ok(ExpertIoReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        phase_gate: "5A-expert-block-io-feasibility".to_string(),
        file: file.display().to_string(),
        file_size_bytes: reader.file_size(),
        os: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        block_bytes,
        requests_per_run: requests,
        repetitions: runs,
        seed,
        page_cache_controlled: false,
        note: "Portable seek/read baseline. OS page cache is not controlled; repeated runs may be warm. This measures storage access only, not end-to-end model tokens/s.".to_string(),
        patterns: vec![sequential, random],
    })
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
    if requests == 0 || cache_mib == 0 {
        return Err("requests and cache-mib must be greater than zero".into());
    }
    if trace_file.is_none() && !["phase-shift", "stationary-hot", "uniform", "round-robin"].contains(&workload) {
        return Err(format!("unsupported workload: {workload}").into());
    }
    let file_size = std::fs::metadata(file)?.len();
    let (table, block_bytes_report) = if let Some(index_path) = index_file {
        let contents = std::fs::read(index_path)?;
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
        let input: ExpertTraceInput = serde_json::from_slice(&std::fs::read(trace_path)?)?;
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
) -> Result<PrefetchReplayReport, Box<dyn std::error::Error>> {
    use std::collections::HashSet;

    if block_bytes == 0 || top_k == 0 || max_pending == 0 || max_inflight_mib == 0 {
        return Err("block-bytes, top-k, max-pending and max-inflight-mib must be greater than zero".into());
    }
    let input: ExpertTraceInput = serde_json::from_slice(&std::fs::read(trace_file)?)?;
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
    let mut candidates: Vec<(ExpertKey, u64)> = distinct.iter()
        .copied()
        .map(|key| (key, block_bytes))
        .collect();
    candidates.sort_by_key(|(key, _)| (key.layer, key.expert_id));
    let distinct_experts_in_trace = candidates.len();
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
        trace_events: trace.len(),
        distinct_experts_in_trace,
        block_bytes,
        top_k,
        max_pending,
        max_inflight_bytes,
        simulated_latency_requests: latency_requests,
        next_event_coverage,
        next_event_precision,
        predicted_candidates,
        correct_next_event_predictions,
        stats,
        note: "Trace replay only. Latency is simulated in trace-access steps; no disk reads, live routing capture, or compute/I/O overlap are performed. Trace entries may be flat IDs (layer 0), [layer, expert_id] pairs, or {layer, expert_id} objects; fixed block sizes are assumed. Candidate universe is limited to experts present in the supplied trace. Compare diagnostic metrics with a no-prefetch baseline; do not treat them as inference performance.".to_string(),
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
    println!("  {}", report.note);
}

fn print_expert_cache_human(report: &ExpertCacheReport) {
    println!("KILN expert-cache trace replay");
    println!("  phase gate: {} | file: {} ({} bytes)", report.phase_gate, report.file, report.file_size_bytes);
    println!("  block: {} bytes | experts: {} | requests: {} | workload: {} | seed: {}",
        report.block_bytes, report.expert_blocks_available, report.requests, report.workload, report.seed);
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
    println!("  host:           {} / {}", report.os, report.architecture);
    println!("  block size:     {} bytes", report.block_bytes);
    println!("  requests/run:   {}", report.requests_per_run);
    println!("  repetitions:    {}", report.repetitions);
    println!("  random seed:    {}", report.seed);
    println!("  page cache:     uncontrolled");
    for pattern in &report.patterns {
        println!("");
        println!("  {} access", pattern.access_pattern);
        println!("    throughput:   {:.2} MiB/s", pattern.throughput_mib_s);
        println!("    latency p50:  {:.2} us", pattern.request_latency_p50_us);
        println!("    latency p95:  {:.2} us", pattern.request_latency_p95_us);
        println!("    total read:   {} bytes", pattern.total_bytes_read);
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
        Commands::PrefetchReplay { trace_file, block_bytes, top_k, max_pending, max_inflight_mib, latency_requests, json } => {
            match run_prefetch_replay(&trace_file, block_bytes, top_k, max_pending, max_inflight_mib, latency_requests) {
                Ok(report) if json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                Ok(report) => print_prefetch_replay_human(&report),
                Err(error) => {
                    eprintln!("prefetch replay failed: {error}");
                    std::process::exit(2);
                }
            }
        }
        Commands::ExpertIo { file, block_bytes, requests, runs, seed, json } => {
            match run_expert_io(&file, block_bytes, requests, runs, seed) {
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
        assert!(report.patterns.iter().all(|p| p.throughput_mib_s > 0.0));
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
            &path, 1024, 999, 1, 42, "uniform", Some(&trace_path), None
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
            &path, 1024, 10, 1, 42, "uniform", Some(&trace_path)
        ).is_err());

        std::fs::remove_file(trace_path).unwrap();
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
        let report = run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0).unwrap();
        assert_eq!(report.trace_events, 8);
        assert_eq!(report.distinct_experts_in_trace, 2);
        std::fs::write(&trace_path, "[[0,1],[1,1],[0,1],[1,2],[0,1]]").unwrap();
        let layered = run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0).unwrap();
        assert_eq!(layered.trace_events, 5);
        assert_eq!(layered.distinct_experts_in_trace, 3);
        assert!(report.next_event_coverage > 0.0);
        assert!(report.predicted_candidates > 0);
        assert_eq!(report.stats.scheduled, report.stats.useful + report.stats.late + report.stats.unused + report.stats.cancelled);
        std::fs::write(&trace_path, "[]").unwrap();
        assert!(run_prefetch_replay(&trace_path, 1024, 1, 4, 1, 0).is_err());
        std::fs::remove_file(trace_path).unwrap();
    }

    #[test]
    fn expert_cache_replay_rejects_invalid_inputs() {
        let path = fixture();
        assert!(run_expert_cache(&path, 0, 10, 1, 42, "uniform", None).is_err());
        assert!(run_expert_cache(&path, 1024, 10, 0, 42, "uniform", None, None).is_err());
        assert!(run_expert_cache(&path, 1024, 10, 1, 42, "unknown", None, None).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
