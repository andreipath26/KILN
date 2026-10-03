//! KILN command-line interface.
//!
//! Seven commands: serve, run, pull, list, bench, plan, info.
//! For Phase 0B, serve and info do real work. The rest print a
//! not-yet-implemented message and exit cleanly.

use clap::{Parser, Subcommand};

use kiln_core::{LinuxMonitor, PerformanceMonitor};
use kiln_core::pipeline::{run_once as pipeline_run_once, Loader};
use kiln_core::chat::{ChatSession, Sampler, SamplingStrategy, SyntheticForward};
use kiln_core::transformer_config::TransformerConfig;
use kiln_models::tokenizer::BpeTokenizer;
use kiln_models::gguf::GgufFile;
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
    /// Print the transformer config read from a GGUF file.
    Config {
        /// Path to the GGUF file.
        model: String,
    },
    /// Tokenize a string using a model's tokenizer.
    Tokenize {
        /// Path to the GGUF file.
        #[arg(long)]
        model: String,
        /// The text to tokenize.
        text: String,
        /// Also decode the tokens back to text.
        #[arg(long)]
        roundtrip: bool,
    },
    /// Inspect a GGUF file: print metadata and tensor table.
    Inspect {
        /// Path to the GGUF file.
        path: String,
        /// Print every tensor. Without this flag, print only the count.
        #[arg(long)]
        tensors: bool,
    },
    /// Start an interactive chat session. Uses a synthetic forward
    /// pass until the real transformer lands.
    Chat {
        /// Path to a GGUF file with tokenizer metadata.
        #[arg(long)]
        model: String,
        /// Random seed for deterministic sampling.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Maximum new tokens per response.
        #[arg(long, default_value_t = 50)]
        max_tokens: usize,
        /// Sampling strategy: greedy, topk, or topp.
        #[arg(long, default_value = "greedy")]
        strategy: String,
    },
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
        Commands::Config { model } => {
            let path_buf = std::path::PathBuf::from(&model);
            let g = match GgufFile::open(&path_buf) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("failed to open {}: {}", model, e);
                    std::process::exit(1);
                }
            };
            match TransformerConfig::from_gguf(&g) {
                Ok(cfg) => {
                    println!("KILN transformer config");
                    println!("  arch:               {}", cfg.arch);
                    println!("  vocab_size:         {}", cfg.vocab_size);
                    println!("  hidden_size:        {}", cfg.hidden_size);
                    println!("  num_layers:         {}", cfg.num_layers);
                    println!("  intermediate_size:  {}", cfg.intermediate_size);
                    println!("  num_heads:          {}", cfg.num_heads);
                    println!("  num_kv_heads:       {}", cfg.num_kv_heads);
                    println!("  head_dim:           {}", cfg.head_dim());
                    println!("  rms_norm_eps:       {}", cfg.rms_norm_eps);
                    println!("  context_length:     {}", cfg.context_length);
                    println!("  rope_theta:         {}", cfg.rope_theta);
                }
                Err(e) => {
                    eprintln!("config error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Tokenize { model, text, roundtrip } => {
            let path_buf = std::path::PathBuf::from(&model);
            let g = match GgufFile::open(&path_buf) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("failed to open {}: {}", model, e);
                    std::process::exit(1);
                }
            };
            let tok = match BpeTokenizer::from_gguf(&g) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("failed to build tokenizer: {}", e);
                    std::process::exit(1);
                }
            };
            println!("vocab_size: {}", tok.vocab_size());
            let ids = tok.encode(&text);
            println!("input:      {:?}", text);
            println!("token_ids:  {:?}", ids);
            println!("token_count: {}", ids.len());
            for (i, &id) in ids.iter().enumerate() {
                let s = tok.token_to_str(id).unwrap_or("<unknown>");
                println!("  [{}] id={} str={:?}", i, id, s);
            }
            if roundtrip {
                let decoded = tok.decode(&ids);
                println!("decoded:    {:?}", decoded);
                println!("roundtrip_ok: {}", decoded == text);
            }
        }
        Commands::Inspect { path, tensors } => {
            let path_buf = std::path::PathBuf::from(&path);
            let g = match GgufFile::open(&path_buf) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("failed to open {}: {}", path, e);
                    std::process::exit(1);
                }
            };
            println!("KILN inspect: {}", path);
            println!("  version:        {}", g.header.version);
            println!("  tensor_count:   {}", g.header.tensor_count);
            println!("  metadata_count: {}", g.header.metadata_kv_count);
            println!("  data_offset:    {}", g.data_offset);
            println!("  file_size:      {}", g.file_size());
            println!();
            println!("Metadata:");
            let mut keys: Vec<&String> = g.metadata.keys().collect();
            keys.sort();
            for k in keys {
                let v = &g.metadata[k];
                let s = format!("{:?}", v);
                let short = if s.len() > 100 { format!("{}...", &s[..100]) } else { s };
                println!("  {} = {}", k, short);
            }
            println!();
            if tensors {
                println!("Tensors ({} total):", g.tensors.len());
                for t in &g.tensors {
                    println!("  {:<40} {:>20} {:?} offset={}",
                        t.name,
                        t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"),
                        t.dtype,
                        t.offset
                    );
                }
            } else {
                println!("Tensors: {} (use --tensors to list)", g.tensors.len());
                // Print dtype distribution.
                use std::collections::HashMap;
                let mut dtypes: HashMap<String, usize> = HashMap::new();
                for t in &g.tensors {
                    let key = format!("{:?}", t.dtype);
                    *dtypes.entry(key).or_insert(0) += 1;
                }
                println!("Dtype distribution:");
                let mut pairs: Vec<(&String, &usize)> = dtypes.iter().collect();
                pairs.sort_by(|a, b| b.1.cmp(a.1));
                for (k, v) in pairs {
                    println!("  {:<20} {}", k, v);
                }
            }
        }
        Commands::Chat { model, seed, max_tokens, strategy } => {
            // Load the GGUF file.
            let path_buf = std::path::PathBuf::from(&model);
            let g = match GgufFile::open(&path_buf) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("failed to open {}: {}", model, e);
                    std::process::exit(1);
                }
            };

            // Build the tokenizer from the GGUF metadata.
            let tokenizer = match BpeTokenizer::from_gguf(&g) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("failed to build tokenizer: {}", e);
                    std::process::exit(1);
                }
            };

            let vocab_size = tokenizer.vocab_size();
            println!("KILN chat session");
            println!("  model:     {}", model);
            println!("  vocab:     {} tokens", vocab_size);
            println!("  seed:      {}", seed);
            println!("  max_tokens: {}", max_tokens);
            println!("  strategy:  {}", strategy);
            println!("  forward:   synthetic (real transformer not yet wired)");
            println!();
            println!("Type a message and press Enter. Ctrl-D to exit.");
            println!();

            // Build the sampling strategy.
            let strat = match strategy.as_str() {
                "greedy" => SamplingStrategy::Greedy,
                "topk" => SamplingStrategy::TopK { k: 40, temperature: 0.8 },
                "topp" => SamplingStrategy::TopP { p: 0.9, temperature: 0.8 },
                other => {
                    eprintln!("unknown strategy: {} (expected greedy, topk, or topp)", other);
                    std::process::exit(2);
                }
            };

            let forward = Box::new(SyntheticForward::new(vocab_size, seed));
            let sampler = Sampler::new(strat, seed);
            let eos = tokenizer.eos_token_id;
            let mut session = ChatSession::new(tokenizer, forward, sampler, max_tokens, eos);

            // Read lines from stdin and generate responses.
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                print!("> ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("input error: {}", e);
                        break;
                    }
                }
                let prompt = line.trim_end();
                if prompt.is_empty() {
                    continue;
                }
                match session.generate(prompt) {
                    Ok(response) => {
                        println!("{}", response);
                    }
                    Err(e) => {
                        eprintln!("chat error: {}", e);
                    }
                }
            }
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
