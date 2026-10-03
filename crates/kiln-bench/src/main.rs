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
