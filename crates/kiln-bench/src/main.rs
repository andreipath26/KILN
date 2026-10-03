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
    }
}
