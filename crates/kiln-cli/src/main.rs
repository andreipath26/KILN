//! KILN command-line interface.
//!
//! Seven commands: serve, run, pull, list, bench, plan, info.
//! For Phase 0B, serve and info do real work. The rest print a
//! not-yet-implemented message and exit cleanly.

use clap::{Parser, Subcommand};

use kiln_core::{LinuxMonitor, PerformanceMonitor};
use kiln_core::pipeline::{run_once as pipeline_run_once, Loader};
use kiln_api::serve;

/// KILN. The fastest local LLM runtime on Earth.
#[derive(Parser, Debug)]
#[command(name = "kiln", version, about = "Fastest local LLM runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the Ollama-compatible REST API server.
    Serve {
        /// Port to bind to. Default 11435.
        #[arg(long, default_value_t = 11435)]
        port: u16,
        /// Host to bind to. Default 127.0.0.1.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Start an interactive chat with a model. Not yet implemented.
    Run { model: String },
    /// Download a model from the catalog. Not yet implemented.
    Pull { model: String },
    /// List installed models. Not yet implemented.
    List,
    /// Run the benchmark suite. Not yet implemented.
    Bench { model: String },
    /// Print the execution plan for a model without running it. Not yet implemented.
    Plan { model: String },
    /// Print the hardware profile.
    Info,
    /// Run the pipeline once on a synthetic model file.
    Pipeline {
        /// Path to the model file.
        path: String,
        /// Which loader to use: synthetic or gguf.
        #[arg(long, default_value = "synthetic")]
        loader: String,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
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
        Commands::Serve { port, host } => {
            if let Err(e) = serve(&host, port).await {
                eprintln!("server error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Info => {
            let mut m = LinuxMonitor::new();
            m.sample();
            let env = m.current_envelope();
            println!("KILN hardware profile");
            println!("  version:                {}", env!("CARGO_PKG_VERSION"));
            println!("  ram_total_bytes:        {}", m.ram_total_bytes());
            println!("  ram_total_gb:           {:.2}", m.ram_total_bytes() as f64 / 1_073_741_824.0);
            println!("  ram_headroom_bytes:     {}", env.ram_headroom_bytes);
            println!("  ram_headroom_gb:        {:.2}", env.ram_headroom_bytes as f64 / 1_073_741_824.0);
            println!("  thermal_state:          {:?}", env.thermal_state);
            println!("  throughput_fraction:    {:.3}", env.throughput_fraction);
        }
        Commands::Pipeline { path, loader, json } => {
            let path_buf = std::path::PathBuf::from(&path);
            let ld = match loader.as_str() {
                "gguf" => Loader::Gguf,
                "synthetic" => Loader::Synthetic,
                other => {
                    eprintln!("unknown loader: {} (expected synthetic or gguf)", other);
                    std::process::exit(2);
                }
            };
            match pipeline_run_once(path_buf, ld) {
                Ok(report) => {
                    if json {
                        println!("{{\"output\":{},\"load_time_nanos\":{},\"select_time_nanos\":{},\"schedule_time_nanos\":{},\"execute_time_nanos\":{},\"total_time_nanos\":{},\"nodes_executed\":{}}}",
                            report.output,
                            report.load_time_nanos,
                            report.select_time_nanos,
                            report.schedule_time_nanos,
                            report.execute_time_nanos,
                            report.total_time_nanos,
                            report.nodes_executed,
                        );
                    } else {
                        println!("KILN pipeline report");
                        println!("  output:               {}", report.output);
                        println!("  nodes_executed:       {}", report.nodes_executed);
                        println!("  load_time_nanos:      {}", report.load_time_nanos);
                        println!("  select_time_nanos:    {}", report.select_time_nanos);
                        println!("  schedule_time_nanos:  {}", report.schedule_time_nanos);
                        println!("  execute_time_nanos:   {}", report.execute_time_nanos);
                        println!("  total_time_nanos:     {}", report.total_time_nanos);
                    }
                }
                Err(e) => {
                    eprintln!("pipeline error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Run { model } => not_implemented("run", &model),
        Commands::Pull { model } => not_implemented("pull", &model),
        Commands::List => not_implemented("list", ""),
        Commands::Bench { model } => not_implemented("bench", &model),
        Commands::Plan { model } => not_implemented("plan", &model),
    }
}

fn not_implemented(cmd: &str, arg: &str) -> ! {
    if arg.is_empty() {
        eprintln!("kiln {}: not yet implemented in Phase 0B", cmd);
    } else {
        eprintln!("kiln {} {}: not yet implemented in Phase 0B", cmd, arg);
    }
    std::process::exit(2);
}
