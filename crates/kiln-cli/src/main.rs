//! KILN command-line interface.
//!
//! Seven commands: serve, run, pull, list, bench, plan, info.
//! For Phase 0B, serve and info do real work. The rest print a
//! not-yet-implemented message and exit cleanly.

use clap::{Parser, Subcommand};

use kiln_core::{LinuxMonitor, PerformanceMonitor};
use kiln_core::pipeline::{run_once as pipeline_run_once, Loader};
use kiln_core::chat::{ChatSession, Sampler, SamplingStrategy};
use kiln_core::transformer_config::TransformerConfig;
use kiln_core::transformer::Transformer;
use kiln_core::transformer_weights::TransformerWeights;
use kiln_models::tokenizer::BpeTokenizer;
use kiln_models::gguf::GgufFile;
use kiln_api::serve;

mod llama_forward;

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
    /// Load every weight tensor from a GGUF file and report sizes.
    Weights {
        /// Path to the GGUF file.
        model: String,
    },
    /// Print the transformer config read from a GGUF file.
    Config {
        /// Path to the GGUF file.
        model: String,
    },
    /// Print the top-N logits for a prompt without generating.
    Debug {
        /// Path to the GGUF file.
        model: String,
        /// The prompt.
        prompt: String,
        /// Number of top tokens to print.
        #[arg(long, default_value_t = 20)]
        top: usize,
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
    /// Run the pipeline once on a GGUF model file.
    Pipeline {
        /// Path to the GGUF model file.
        path: String,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Convert a GGUF model to a different quantization.
    Convert {
        /// Input GGUF path.
        #[arg(long)]
        input: String,
        /// Output GGUF path.
        #[arg(long)]
        output: String,
        /// Target quantization. Only tq1_0 is supported.
        #[arg(long, default_value = "tq1_0")]
        quant: String,
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
        Commands::Weights { model } => {
            let path_buf = std::path::PathBuf::from(&model);
            let g = match GgufFile::open(&path_buf) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("failed to open {}: {}", model, e);
                    std::process::exit(1);
                }
            };
            let g = std::sync::Arc::new(g);
            match TransformerWeights::from_gguf(&g) {
                Ok(w) => {
                    println!("KILN transformer weights");
                    println!("  config:      {}", w.config.summary());
                    println!("  token_embd:  {} bytes", w.token_embd.len());
                    println!("  output_norm: {} bytes", w.output_norm.len());
                    println!("  output:      {} bytes", w.output.len());
                    println!("  layers:      {}", w.layers.len());
                    let l = &w.layers[0];
                    println!("  layer 0:");
                    println!("    attn_norm:   {} bytes", l.attn_norm.len());
                    println!("    attn_q:      {} bytes", l.attn_q.len());
                    println!("    attn_k:      {} bytes", l.attn_k.len());
                    println!("    attn_v:      {} bytes", l.attn_v.len());
                    println!("    attn_output: {} bytes", l.attn_output.len());
                    println!("    ffn_norm:    {} bytes", l.ffn_norm.len());
                    println!("    ffn_gate:    {} bytes", l.ffn_gate.len());
                    println!("    ffn_up:      {} bytes", l.ffn_up.len());
                    println!("    ffn_down:    {} bytes", l.ffn_down.len());
                    let total = w.total_bytes();
                    println!("  total:       {} bytes ({:.2} GB)",
                        total, total as f64 / 1_073_741_824.0);
                }
                Err(e) => {
                    eprintln!("weights error: {}", e);
                    std::process::exit(1);
                }
            }
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
        Commands::Debug { model, prompt, top } => {
            let path = std::path::PathBuf::from(&model);
            let mut ctx = match kiln_runtime::LlamaContext::load(&path) {
                Ok(c) => c,
                Err(e) => { eprintln!("load: {}", e); std::process::exit(1); }
            };
            println!("prompt: {:?}", prompt);
            let ids = match ctx.tokenize(&prompt) {
                Ok(t) => t,
                Err(e) => { eprintln!("tokenize: {}", e); std::process::exit(1); }
            };
            println!("tokens: {:?}", ids);
            let t0 = std::time::Instant::now();
            let logits = match ctx.forward(&ids) {
                Ok(l) => l,
                Err(e) => { eprintln!("forward: {}", e); std::process::exit(1); }
            };
            let dt = t0.elapsed().as_secs_f64();
            println!("forward: {:.4}s", dt);
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            println!("top {}:", top);
            for k in 0..top.min(idx.len()) {
                let i = idx[k];
                let s = ctx.token_to_str(i as i32);
                println!("  {:>6}  {:>10.4}  {:?}", i, logits[i], s);
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
        Commands::Convert { input, output, quant } => {
            if quant != "tq1_0" {
                eprintln!("convert: only tq1_0 is supported, got {}", quant);
                std::process::exit(2);
            }
            let inp = std::path::PathBuf::from(&input);
            let outp = std::path::PathBuf::from(&output);
            match kiln_models::convert_to_tq1_0(&inp, &outp) {
                Ok(n) => println!("converted {} tensors -> {}", n, output),
                Err(e) => { eprintln!("convert error: {}", e); std::process::exit(1); }
            }
        }
        Commands::Chat { model, seed, max_tokens, strategy } => {
            use kiln_core::chat::{Sampler, SamplingStrategy};

            let path = std::path::PathBuf::from(&model);
            let mut ctx = match kiln_runtime::LlamaContext::load(&path) {
                Ok(c) => c,
                Err(e) => { eprintln!("load: {}", e); std::process::exit(1); }
            };
            let vocab_size = ctx.n_vocab();
            let eos = ctx.eos_token();
            println!("KILN chat session (Phase 2.3, multi-turn)");
            println!("  model:      {}", model);
            println!("  vocab:      {} tokens", vocab_size);
            println!("  eos:        {}", eos);
            println!("  seed:       {}", seed);
            println!("  max_tokens: {}", max_tokens);
            println!("  strategy:   {}", strategy);
            println!("  runtime:    llama.cpp via kiln-runtime");
            println!();
            println!("Type a message and press Enter. Ctrl-D to exit.");
            println!();

            let strat = match strategy.as_str() {
                "greedy" => SamplingStrategy::Greedy,
                "topk" => SamplingStrategy::TopK { k: 40, temperature: 0.8 },
                "topp" => SamplingStrategy::TopP { p: 0.9, temperature: 0.8 },
                other => {
                    eprintln!("unknown strategy: {} (expected greedy, topk, or topp)", other);
                    std::process::exit(2);
                }
            };
            let mut sampler = Sampler::new(strat, seed);

            // Multi-turn history. Each entry is (user, assistant).
            let mut history: Vec<(String, String)> = Vec::new();

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
                    Err(e) => { eprintln!("input error: {}", e); break; }
                }
                let prompt = line.trim_end();
                if prompt.is_empty() { continue; }

                // Rebuild the full ChatML prompt with history.
                let mut templated = String::new();
                templated.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
                for (u, a) in &history {
                    templated.push_str("<|im_start|>user\n");
                    templated.push_str(u);
                    templated.push_str("<|im_end|>\n<|im_start|>assistant\n");
                    templated.push_str(a);
                    templated.push_str("<|im_end|>\n");
                }
                templated.push_str("<|im_start|>user\n");
                templated.push_str(prompt);
                templated.push_str("<|im_end|>\n<|im_start|>assistant\n");

                // Reset the KV cache so the prompt can be re-prefilled
                // with the full history from position 0.
                if let Err(e) = ctx.reset() {
                    eprintln!("reset: {}", e);
                    continue;
                }

                let t0 = std::time::Instant::now();
                let prompt_ids = match ctx.tokenize(&templated) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("tokenize: {}", e); continue; }
                };
                let mut next_logits = match ctx.forward(&prompt_ids) {
                    Ok(l) => l,
                    Err(e) => { eprintln!("forward: {}", e); continue; }
                };

                let mut reply = String::new();
                let mut n_gen = 0usize;
                for _ in 0..max_tokens {
                    let token = match sampler.sample(&next_logits) {
                        Some(t) => t,
                        None => break,
                    };
                    if (token as i32) == eos { break; }
                    let piece = ctx.token_to_str(token as i32);
                    if piece.is_empty() { break; }
                    print!("{}", piece);
                    std::io::stdout().flush().ok();
                    reply.push_str(&piece);
                    n_gen += 1;
                    match ctx.forward(&[token as i32]) {
                        Ok(l) => next_logits = l,
                        Err(e) => { eprintln!("\nforward: {}", e); break; }
                    }
                }
                println!();
                let dt = t0.elapsed().as_secs_f64().max(1e-6);
                eprintln!("[{:.2}s, {:.2} tok/s, {} tokens]", dt, n_gen as f64 / dt, n_gen);

                history.push((prompt.to_string(), reply));
            }
        }
        Commands::Pipeline { path, json } => {
            let path_buf = std::path::PathBuf::from(&path);
            match pipeline_run_once(path_buf, Loader::Gguf) {
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
