//! KILN command-line interface.
//!
//! Command-line interface for local KILN inference and model management.

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
mod model_library;
mod phase35_kv_bench;
mod phase35_spec_bench;

/// Linux process RSS snapshot in KiB. VmHWM is the process high-water mark
/// since launch, so it includes the entire model load/benchmark lifetime.
/// This does NOT include page cache held outside process RSS.
#[derive(Debug, Clone, Copy)]
struct ProcessMemoryKib {
    rss_kib: u64,
    peak_rss_kib: u64,
}

fn parse_process_memory_status(status: &str) -> Option<ProcessMemoryKib> {
    fn field_kib(status: &str, key: &str) -> Option<u64> {
        let line = status.lines().find(|line| line.split_once(':')
            .map(|(name, _)| name == key).unwrap_or(false))?;
        let (_, value) = line.split_once(':')?;
        let mut parts = value.split_whitespace();
        let amount = parts.next()?.parse::<u64>().ok()?;
        if parts.next()? != "kB" || parts.next().is_some() {
            return None;
        }
        Some(amount)
    }

    Some(ProcessMemoryKib {
        rss_kib: field_kib(status, "VmRSS")?,
        peak_rss_kib: field_kib(status, "VmHWM")?,
    })
}

fn process_memory_kib() -> Option<ProcessMemoryKib> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| parse_process_memory_status(&status))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod process_memory_tests {
    use super::*;

    #[test]
    fn reads_exact_rss_keys_and_units() {
        let status = "Name:\tkiln\nVmPeak:\t9999 kB\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n";
        let parsed = parse_process_memory_status(status).unwrap();
        assert_eq!(parsed.rss_kib, 1024);
        assert_eq!(parsed.peak_rss_kib, 2048);
        assert!(parse_process_memory_status("VmRSS:\t1024 kB\n").is_none());
        assert!(parse_process_memory_status("VmHWM:\t5 MB\nVmRSS:\t2 kB\n").is_none());
        assert!(parse_process_memory_status("NotVmHWM:\t5 kB\nVmRSS:\t2 kB\n").is_none());
    }
}

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
        /// Loopback IP to bind to. Only 127.0.0.1 or ::1 is accepted; remote access is unsupported.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Start an interactive chat session using the llama.cpp runtime.
    Run {
        /// Path to the GGUF model file.
        model: String,
        /// Maximum tokens generated per response.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Preserve executed KV tokens between turns; uses a generic text dialogue format.
        #[arg(long)]
        keep_context: bool,
        /// Experimental greedy fast path: bypass vocabulary-sized logits transfers.
        #[arg(long, conflicts_with = "spec_type")]
        native_greedy: bool,
        /// Experimental exact n-gram speculative decode mode.
        #[arg(long, value_parser = ["ngram-mod"])]
        spec_type: Option<String>,
    },
    /// Controlled Phase 3.5 second-turn prompt-cache A/B experiment.
    BenchKv {
        /// Local GGUF file; never downloads model weights.
        model: String,
        #[arg(long, default_value = "Name two planets.")]
        first_prompt: String,
        #[arg(long, default_value = "Name one more.")]
        second_prompt: String,
        #[arg(long, default_value = " Mars and Venus.")]
        assistant_fragment: String,
        #[arg(long, default_value_t = 5)]
        runs: usize,
        #[arg(long)]
        threads: Option<u32>,
        #[arg(long)]
        batch: Option<u32>,
        #[arg(long)]
        ubatch: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// A/B test draftless n-gram speculative decoding with exact target-model verification.
    BenchSpec {
        /// Pinned GGUF; no download or external draft model required.
        model: String,
        #[arg(long, default_value = "Repeat this short phrase: red blue red blue red blue.")]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        #[arg(long, default_value_t = 5)]
        runs: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long)]
        threads: Option<u32>,
        #[arg(long)]
        batch: Option<u32>,
        #[arg(long)]
        ubatch: Option<u32>,
        #[arg(long, default_value_t = 3)]
        ngram: usize,
        #[arg(long, default_value_t = 8)]
        draft_max: usize,
        #[arg(long)]
        json: bool,
    },
    /// Download and validate a GGUF file from a public Hugging Face repository.
    Pull {
        /// Repository in owner/name form, as shown by `kiln search`.
        model: String,
        /// Exact GGUF file path inside that repository.
        #[arg(long)]
        file: String,
    },
    /// Find readable GGUF model candidates already installed on this machine.
    List {
        /// Additional file or directory to scan. May be repeated.
        #[arg(long = "path")]
        paths: Vec<std::path::PathBuf>,
        /// Emit JSON instead of human-readable results.
        #[arg(long)]
        json: bool,
    },
    /// Search public Hugging Face repositories tagged for GGUF models.
    Search {
        /// Search phrase or model family, for example `granite 4.1`.
        query: String,
        /// Maximum repositories to inspect (1–50).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Measure model load, prompt prefill, and token decode throughput.
    Bench {
        /// Path to the GGUF model file.
        model: String,
        /// Prompt used for every measured run.
        #[arg(long, default_value = "The meaning of life is")]
        prompt: String,
        /// Maximum number of generated tokens per run.
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        /// Number of measured runs (reported individually and as a mean).
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Number of unreported warm-up runs before measurement.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Override the detected CPU thread count for controlled comparisons.
        #[arg(long)]
        threads: Option<u32>,
        /// Override the detected batch size for controlled comparisons.
        #[arg(long)]
        batch: Option<u32>,
        /// Physical microbatch size for controlled comparisons.
        #[arg(long)]
        ubatch: Option<u32>,
        /// Compare native greedy sampling against Rust vocabulary logits sampling.
        #[arg(long)]
        native_greedy: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
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
    /// Validate a model: load checks and canary prompts.
    Doctor {
        /// Path to the GGUF file.
        model: String,
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
                    if let Some(l) = w.layers.first() {
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
                    } else {
                        println!("  layer details: unavailable (model has no transformer layers)");
                    }
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
            use kiln_runtime::{RuntimeDispatcher, RuntimeOp, RuntimeResult};
            use kiln_runtime::LlamaCppBackend;

            let path = std::path::PathBuf::from(&model);
            let prof = kiln_runtime::SystemProfile::load_or_probe();
            let mut d = RuntimeDispatcher::new();
            d.register(Box::new(LlamaCppBackend::new()));
            if let Err(e) = d.dispatch(RuntimeOp::Load {
                path,
                n_ctx: prof.n_ctx,
                n_batch: prof.n_batch,
                n_ubatch: prof.n_batch.min(512),
                n_threads: prof.n_threads,
            }) {
                eprintln!("load: {}", e);
                std::process::exit(1);
            }
            println!("prompt: {:?}", prompt);
            let ids = match d.dispatch(RuntimeOp::Tokenize { text: prompt.clone() }) {
                Ok(RuntimeResult::Tokens(t)) => t,
                Ok(_) => { eprintln!("tokenize: wrong return"); std::process::exit(1); }
                Err(e) => { eprintln!("tokenize: {}", e); std::process::exit(1); }
            };
            println!("tokens: {:?}", ids);
            let t0 = std::time::Instant::now();
            let logits = match d.dispatch(RuntimeOp::Forward { tokens: ids }) {
                Ok(RuntimeResult::Logits(l)) => l,
                Ok(_) => { eprintln!("forward: wrong return"); std::process::exit(1); }
                Err(e) => { eprintln!("forward: {}", e); std::process::exit(1); }
            };
            let dt = t0.elapsed().as_secs_f64();
            println!("forward: {:.4}s", dt);
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            println!("top {}:", top);
            for k in 0..top.min(idx.len()) {
                let i = idx[k];
                let tokstr = match d.dispatch(RuntimeOp::TokenToStr { token: i as i32 }) {
                    Ok(RuntimeResult::Bytes(b)) => String::from_utf8_lossy(&b).to_string(),
                    Ok(RuntimeResult::Str(s)) => s,
                    _ => String::new(),
                };
                println!("  {:>6}  {:>10.4}  {:?}", i, logits[i], tokstr);
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
                let short = truncate_for_display(&s, 100);
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
            use kiln_runtime::{RuntimeDispatcher, RuntimeOp, RuntimeResult, LlamaCppBackend};

            let path = std::path::PathBuf::from(&model);
            let prof = kiln_runtime::SystemProfile::load_or_probe();
            let mut d = RuntimeDispatcher::new();
            d.register(Box::new(LlamaCppBackend::new()));
            if let Err(e) = d.dispatch(RuntimeOp::Load {
                path,
                n_ctx: prof.n_ctx,
                n_batch: prof.n_batch,
                n_ubatch: prof.n_batch.min(512),
                n_threads: prof.n_threads,
            }) {
                eprintln!("load: {}", e);
                std::process::exit(1);
            }
            let vocab_size = match d.dispatch(RuntimeOp::NVocab) {
                Ok(RuntimeResult::Int(n)) => n,
                _ => { eprintln!("n_vocab failed"); std::process::exit(1); }
            };
            let eos = match d.dispatch(RuntimeOp::EosToken) {
                Ok(RuntimeResult::Int(n)) => n,
                _ => { eprintln!("eos_token failed"); std::process::exit(1); }
            };
            println!("KILN chat session (Phase 3, runtime dispatch)");
            println!("  model:      {}", model);
            println!("  vocab:      {} tokens", vocab_size);
            println!("  eos:        {}", eos);
            println!("  seed:       {}", seed);
            println!("  max_tokens: {}", max_tokens);
            println!("  strategy:   {}", strategy);
            println!("  runtime:    llama.cpp via RuntimeDispatcher");
            println!("  profile:    tier={:?} n_ctx={} n_batch={} n_threads={}",
                prof.tier, prof.n_ctx, prof.n_batch, prof.n_threads);
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

                if let Err(e) = d.dispatch(RuntimeOp::Reset) {
                    eprintln!("reset: {}", e);
                    continue;
                }

                let t0 = std::time::Instant::now();
                let prompt_ids = match d.dispatch(RuntimeOp::Tokenize { text: templated }) {
                    Ok(RuntimeResult::Tokens(t)) => t,
                    _ => { eprintln!("tokenize failed"); continue; }
                };
                let mut next_logits = match d.dispatch(RuntimeOp::Forward { tokens: prompt_ids }) {
                    Ok(RuntimeResult::Logits(l)) => l,
                    _ => { eprintln!("forward failed"); continue; }
                };

                let mut reply = String::new();
                let mut n_gen = 0usize;
                let mut pending: Vec<u8> = Vec::new();
                let mut watchdog = kiln_runtime::Watchdog::new();
                for _ in 0..max_tokens {
                    let token = match sampler.sample(&next_logits) {
                        Some(t) => t,
                        None => break,
                    };
                    if (token as i32) == eos { break; }
                    let raw = match d.dispatch(RuntimeOp::TokenToStr { token: token as i32 }) {
                        Ok(RuntimeResult::Bytes(b)) => b,
                        Ok(RuntimeResult::Str(s)) => s.into_bytes(),
                        _ => break,
                    };
                    if raw.is_empty() { break; }
                    // Buffer bytes across tokens. A token may carry a
                    // partial multi-byte UTF-8 sequence that the next
                    // token completes. Decode only when the buffer is
                    // valid UTF-8.
                    pending.extend_from_slice(&raw);
                    match std::str::from_utf8(&pending) {
                        Ok(s) => {
                            print!("{}", s);
                            std::io::stdout().flush().ok();
                            reply.push_str(s);
                            pending.clear();
                        }
                        Err(_) => {
                            // Partial sequence. Keep buffering. If the
                            // buffer grows past 8 bytes without ever
                            // being valid, it is not a partial sequence,
                            // it is garbage. Emit it lossy and clear.
                            if pending.len() > 8 {
                                let s = String::from_utf8_lossy(&pending).to_string();
                                print!("{}", s);
                                std::io::stdout().flush().ok();
                                reply.push_str(&s);
                                pending.clear();
                            }
                        }
                    }
                    n_gen += 1;
                    // Watchdog: top-1 logit is the max of next_logits.
                    let top1 = next_logits.iter().cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    match watchdog.observe(token as i32, &raw, top1) {
                        kiln_runtime::WatchdogVerdict::Ok => {}
                        kiln_runtime::WatchdogVerdict::RepeatedOutput => {
                            eprintln!("\n[watchdog] aborting: repeated output");
                            break;
                        }
                        kiln_runtime::WatchdogVerdict::LowConfidence => {
                            eprintln!("\n[watchdog] aborting: low confidence (top1={})", top1);
                            break;
                        }
                        kiln_runtime::WatchdogVerdict::NonTextOutput => {
                            eprintln!("\n[watchdog] aborting: non-text output");
                            break;
                        }
                    }
                    match d.dispatch(RuntimeOp::Forward { tokens: vec![token as i32] }) {
                        Ok(RuntimeResult::Logits(l)) => next_logits = l,
                        _ => { eprintln!("\nforward failed"); break; }
                    }
                }
                println!();
                let dt = t0.elapsed().as_secs_f64().max(1e-6);
                eprintln!("[{:.2}s, {:.2} tok/s, {} tokens]", dt, n_gen as f64 / dt, n_gen);

                history.push((prompt.to_string(), reply));
            }
        }
        Commands::Doctor { model } => {
            use kiln_runtime::{RuntimeDispatcher, RuntimeOp, RuntimeResult, LlamaCppBackend};
            use kiln_runtime::run_canaries;

            let path = std::path::PathBuf::from(&model);
            let prof = kiln_runtime::SystemProfile::load_or_probe();
            println!("KILN doctor: {}", model);
            println!("  profile: tier={:?} n_ctx={} n_batch={} n_threads={}",
                prof.tier, prof.n_ctx, prof.n_batch, prof.n_threads);

            let mut d = RuntimeDispatcher::new();
            d.register(Box::new(LlamaCppBackend::new()));
            match d.dispatch(RuntimeOp::Load {
                path,
                n_ctx: prof.n_ctx,
                n_batch: prof.n_batch,
                n_ubatch: prof.n_batch.min(512),
                n_threads: prof.n_threads,
            }) {
                Ok(_) => println!("  load: PASS"),
                Err(e) => { println!("  load: FAIL ({})", e); std::process::exit(1); }
            }

            // Canaries need a LlamaContext. The backend owns it, so
            // reload through the context directly for the canary pass.
            let path2 = std::path::PathBuf::from(&model);
            let mut ctx = match kiln_runtime::LlamaContext::load_with(
                &path2, Some((prof.n_ctx as i32, prof.n_batch as i32, prof.n_threads as i32))) {
                Ok(c) => c,
                Err(e) => { println!("  canary: SKIP (reload failed: {})", e); std::process::exit(1); }
            };
            match run_canaries(&mut ctx) {
                Ok(()) => println!("  canary: PASS ({}/{})", kiln_runtime::CANARIES.len(), kiln_runtime::CANARIES.len()),
                Err(e) => { println!("  canary: FAIL ({})", e); std::process::exit(1); }
            }
            println!("  verdict: model is usable");
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
        Commands::Run { model, max_tokens, keep_context, native_greedy, spec_type } => {
            if max_tokens == 0 {
                eprintln!("run: --max-tokens must be at least 1");
                std::process::exit(2);
            }
            if native_greedy && spec_type.is_some() {
                eprintln!("run: --native-greedy cannot be combined with --spec-type; n-gram verification already uses the native greedy target path");
                std::process::exit(2);
            }
            let selected_model = if model == "auto" {
                match model_library::resolve_auto_model() {
                    Ok(path) => {
                        println!("Using local GGUF candidate: {}", path.display());
                        path.display().to_string()
                    }
                    Err(error) => { eprintln!("run auto: {error}"); std::process::exit(2); }
                }
            } else { model };
            if let Err(error) = run_interactive(
                &selected_model, max_tokens, keep_context, native_greedy,
                spec_type.as_deref() == Some("ngram-mod"),
            ) {
                eprintln!("run: {}", error);
                std::process::exit(1);
            }
        },
        Commands::BenchSpec { model, prompt, tokens, runs, warmup,
            threads, batch, ubatch, ngram, draft_max, json } => {
            if let Err(error) = phase35_spec_bench::run(
                model, prompt, tokens, runs, warmup, threads, batch, ubatch,
                ngram, draft_max, json
            ) {
                eprintln!("bench-spec: {error}");
                std::process::exit(1);
            }
        },
        Commands::BenchKv { model, first_prompt, second_prompt, assistant_fragment,
            runs, threads, batch, ubatch, json } => {
            if let Err(error) = phase35_kv_bench::run(
                model, first_prompt, second_prompt, assistant_fragment,
                runs, threads, batch, ubatch, json
            ) {
                eprintln!("bench-kv: {error}");
                std::process::exit(1);
            }
        },
        Commands::Pull { model, file } => {
            match tokio::task::spawn_blocking(move || model_library::pull_hugging_face(&model, &file)).await {
                Ok(Ok(path)) => println!("Installed model: {}", path.display()),
                Ok(Err(error)) => { eprintln!("pull: {error}"); std::process::exit(1); }
                Err(error) => { eprintln!("pull worker failed: {error}"); std::process::exit(1); }
            }
        }
        Commands::List { paths, json } => {
            if let Err(error) = model_library::list_models(&paths, json) {
                eprintln!("list: {error}"); std::process::exit(1);
            }
        }
        Commands::Search { query, limit } => {
            match tokio::task::spawn_blocking(move || model_library::search_hugging_face(&query, limit)).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => { eprintln!("search: {error}"); std::process::exit(1); }
                Err(error) => { eprintln!("search worker failed: {error}"); std::process::exit(1); }
            }
        }
        Commands::Bench { model, prompt, tokens, runs, warmup, threads, batch, ubatch, native_greedy, json } => {
            if runs == 0 {
                eprintln!("bench: --runs must be at least 1");
                std::process::exit(2);
            }
            if tokens == 0 {
                eprintln!("bench: --tokens must be at least 1");
                std::process::exit(2);
            }

            use kiln_runtime::{LlamaCppBackend, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};
            let mut profile = SystemProfile::load_or_probe();
            if let Some(threads) = threads {
                if threads == 0 {
                    eprintln!("bench: --threads must be at least 1");
                    std::process::exit(2);
                }
                profile.n_threads = threads;
            }
            if let Some(batch) = batch {
                if batch == 0 {
                    eprintln!("bench: --batch must be at least 1");
                    std::process::exit(2);
                }
                profile.n_batch = batch;
            }
            let microbatch = ubatch.unwrap_or(profile.n_batch.min(512));
            if microbatch == 0 || microbatch > profile.n_batch {
                eprintln!("bench: --ubatch must be 1..=--batch");
                std::process::exit(2);
            }
            let mut dispatcher = RuntimeDispatcher::new();
            dispatcher.register(Box::new(LlamaCppBackend::new()));

            let memory_before_load = process_memory_kib();
            let load_start = std::time::Instant::now();
            if let Err(error) = dispatcher.dispatch(RuntimeOp::Load {
                path: std::path::PathBuf::from(&model),
                n_ctx: profile.n_ctx,
                n_batch: profile.n_batch,
                n_ubatch: microbatch,
                n_threads: profile.n_threads,
            }) {
                eprintln!("bench: model load failed: {}", error);
                std::process::exit(1);
            }
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            let memory_after_load = process_memory_kib();
            let eos = match dispatcher.dispatch(RuntimeOp::EosToken) {
                Ok(RuntimeResult::Int(token)) => token,
                Ok(_) => {
                    eprintln!("bench: backend returned an unexpected EOS-token result");
                    std::process::exit(1);
                }
                Err(error) => {
                    eprintln!("bench: unable to read EOS token: {}", error);
                    std::process::exit(1);
                }
            };

            let mut warmup_completed = 0usize;
            for index in 0..warmup {
                if let Err(error) = bench_once(&mut dispatcher, &prompt, tokens, eos, native_greedy) {
                    eprintln!("bench: warm-up run {} failed: {}", index + 1, error);
                    std::process::exit(1);
                }
                warmup_completed += 1;
            }

            let mut measurements = Vec::with_capacity(runs);
            for index in 0..runs {
                match bench_once(&mut dispatcher, &prompt, tokens, eos, native_greedy) {
                    Ok(mut result) => {
                        // VmHWM is process-wide and monotonic: a useful upper
                        // bound at this checkpoint, not per-run isolated peak.
                        let sample = process_memory_kib();
                        result["process_rss_kib_after_run"] = serde_json::json!(sample.map(|m| m.rss_kib));
                        result["process_peak_rss_kib_to_date"] = serde_json::json!(sample.map(|m| m.peak_rss_kib));
                        measurements.push(result);
                    },
                    Err(error) => {
                        eprintln!("bench: measured run {} failed: {}", index + 1, error);
                        std::process::exit(1);
                    }
                }
            }

            let memory_after_benchmark = process_memory_kib();

            // Report distribution statistics as well as means. A mean alone can
            // hide noisy runs and is insufficient for controlled comparisons.
            let stats = |key: &str| benchmark_stats(&measurements, key);
            let model_path = std::fs::canonicalize(&model)
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| model.clone());
            let model_size_bytes = std::fs::metadata(&model).map(|metadata| metadata.len()).ok();
            let logical_cpus = std::thread::available_parallelism().map(|n| n.get()).ok();
            let output = serde_json::json!({
                "model": model,
                "model_path_canonical": model_path,
                "model_size_bytes": model_size_bytes,
                "prompt": prompt,
                "max_new_tokens": tokens,
                "warmup_runs": warmup_completed,
                "requested_threads": profile.n_threads,
                "requested_batch": profile.n_batch,
                "requested_microbatch": microbatch,
                "sampling_path": if native_greedy { "native_greedy" } else { "rust_logits_greedy" },
                "measured_runs": runs,
                "load_ms": load_ms,
                "process_memory": {
                    "source": if cfg!(target_os = "linux") { "linux_proc_self_status" } else { "unsupported_platform" },
                    "unit": "KiB",
                    "before_load": {
                        "rss": memory_before_load.map(|m| m.rss_kib),
                        "peak_rss_to_date": memory_before_load.map(|m| m.peak_rss_kib)
                    },
                    "after_load": {
                        "rss": memory_after_load.map(|m| m.rss_kib),
                        "peak_rss_to_date": memory_after_load.map(|m| m.peak_rss_kib)
                    },
                    "after_benchmark": {
                        "rss": memory_after_benchmark.map(|m| m.rss_kib),
                        "peak_rss_to_date": memory_after_benchmark.map(|m| m.peak_rss_kib)
                    },
                    "note": "Process RSS and lifetime peak from /proc/self/status. Linux only. Peak is monotonic, not reset per run, and excludes independently cached filesystem pages and other machine processes."
                },
                "environment": {
                    "os": std::env::consts::OS,
                    "architecture": std::env::consts::ARCH,
                    "logical_cpus": logical_cpus
                },
                "profile": {
                    "tier": format!("{:?}", profile.tier),
                    "n_ctx": profile.n_ctx,
                    "n_batch": profile.n_batch,
                    "n_ubatch": microbatch,
                    "n_threads": profile.n_threads
                },
                "statistics": {
                    "prompt_tokens": stats("prompt_tokens"),
                    "generated_tokens": stats("generated_tokens"),
                    "prefill_ms": stats("prefill_ms"),
                    "decode_ms": stats("decode_ms"),
                    "decode_forward_ms": stats("decode_forward_ms"),
                    "sampling_ms": stats("sampling_ms"),
                    "token_conversion_ms": stats("token_conversion_ms"),
                    "decode_tokens_per_second": stats("decode_tokens_per_second"),
                    "decode_forward_tokens_per_second": stats("decode_forward_tokens_per_second"),
                    "response_tokens_per_second_including_prefill": stats("response_tokens_per_second_including_prefill")
                },
                "runs": measurements
            });

            if json {
                println!("{}", serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string()));
            } else {
                println!("KILN benchmark (RuntimeDispatcher / llama.cpp)");
                println!("  model:             {}", model);
                println!("  profile:           {:?}, threads={}, batch={}, microbatch={}, context={}",
                    profile.tier, profile.n_threads, profile.n_batch, microbatch, profile.n_ctx);
                println!("  load:              {:.2} ms", load_ms);
                if let Some(memory) = memory_after_benchmark {
                    println!("  process RSS:       {} KiB (peak to date {} KiB; does not include OS page cache)",
                        memory.rss_kib, memory.peak_rss_kib);
                }
                println!("  prompt:            {:?}", prompt);
                println!("  warm-up runs:      {}", warmup_completed);
                println!("  measured runs:     {}", runs);
                println!("  rate meaning:      sampled rate includes prefill-derived first token; decode forward excludes it");
                println!("  model size:        {} bytes", model_size_bytes.map(|n| n.to_string()).unwrap_or_else(|| "unknown".into()));
                println!("  environment:       {} / {}, logical CPUs={}",
                    std::env::consts::OS, std::env::consts::ARCH,
                    logical_cpus.map(|n| n.to_string()).unwrap_or_else(|| "unknown".into()));
                for (label, key, unit) in [
                    ("prompt tokens", "prompt_tokens", ""),
                    ("prefill", "prefill_ms", " ms"),
                    ("generated", "generated_tokens", " tokens"),
                    ("decode", "decode_ms", " ms"),
                    ("forward time", "decode_forward_ms", " ms"),
                    ("sampling", "sampling_ms", " ms"),
                    ("token conversion", "token_conversion_ms", " ms"),
                    ("sampled rate", "decode_tokens_per_second", " tokens/s"),
                    ("decode forward", "decode_forward_tokens_per_second", " steps/s"),
                    ("including prefill", "response_tokens_per_second_including_prefill", " tokens/s"),
                ] {
                    let summary = stats(key);
                    if let Some(summary) = summary.as_object() {
                        println!("  {:<18} median {:>8.2}{} | mean {:>8.2} | range {:.2}–{:.2}",
                            format!("{}:", label),
                            summary["median"].as_f64().unwrap_or(0.0), unit,
                            summary["mean"].as_f64().unwrap_or(0.0),
                            summary["min"].as_f64().unwrap_or(0.0),
                            summary["max"].as_f64().unwrap_or(0.0));
                    }
                }
                println!();
                for (index, run) in measurements.iter().enumerate() {
                    println!("  run {}: prefill {:.2} ms; decode {:.2} ms; {:.2} tokens/s ({} prompt, {} generated)",
                        index + 1,
                        run["prefill_ms"].as_f64().unwrap_or(0.0),
                        run["decode_ms"].as_f64().unwrap_or(0.0),
                        run["decode_tokens_per_second"].as_f64().unwrap_or(0.0),
                        run["prompt_tokens"].as_u64().unwrap_or(0),
                        run["generated_tokens"].as_u64().unwrap_or(0));
                }
            }
        },
        Commands::Plan { model } => not_implemented("plan", &model),
    }
}

/// Summarize one numeric metric across benchmark runs.
fn benchmark_stats(measurements: &[serde_json::Value], key: &str) -> serde_json::Value {
    let mut values: Vec<f64> = measurements.iter()
        .filter_map(|run| run.get(key).and_then(serde_json::Value::as_f64))
        .collect();
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let median = if count % 2 == 0 {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    } else {
        values[count / 2]
    };
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count as f64;
    serde_json::json!({
        "mean": mean,
        "median": median,
        "min": values[0],
        "max": values[count - 1],
        "stddev_population": variance.sqrt()
    })
}

#[cfg(test)]
mod benchmark_stats_tests {
    use super::benchmark_stats;
    use serde_json::json;

    #[test]
    fn reports_median_range_and_population_spread() {
        let runs = vec![
            json!({"rate": 1.0}),
            json!({"rate": 3.0}),
            json!({"rate": 2.0}),
        ];
        let stats = benchmark_stats(&runs, "rate");
        assert_eq!(stats["mean"].as_f64().unwrap(), 2.0);
        assert_eq!(stats["median"].as_f64().unwrap(), 2.0);
        assert_eq!(stats["min"].as_f64().unwrap(), 1.0);
        assert_eq!(stats["max"].as_f64().unwrap(), 3.0);
    }

    #[test]
    fn even_count_median_averages_middle_values() {
        let runs = vec![json!({"rate": 1.0}), json!({"rate": 3.0})];
        assert_eq!(benchmark_stats(&runs, "rate")["median"].as_f64().unwrap(), 2.0);
    }

    #[test]
    fn missing_metric_returns_null() {
        assert!(benchmark_stats(&[json!({"other": 1})], "rate").is_null());
    }
}

/// Buffer token bytes until complete UTF-8 characters are available.
fn decode_utf8_piece(pending: &mut Vec<u8>, bytes: &[u8]) -> String {
    pending.extend_from_slice(bytes);
    match std::str::from_utf8(pending) {
        Ok(text) => {
            let decoded = text.to_string();
            pending.clear();
            decoded
        }
        Err(error) => {
            let valid = error.valid_up_to();
            let decoded = String::from_utf8_lossy(&pending[..valid]).into_owned();
            if error.error_len().is_some() {
                let remainder = String::from_utf8_lossy(&pending[valid..]).into_owned();
                pending.clear();
                format!("{}{}", decoded, remainder)
            } else {
                let remainder = pending.split_off(valid);
                *pending = remainder;
                decoded
            }
        }
    }
}

/// Validate backend draft-verification metadata before updating history/KV.
fn validate_verified_draft(accepted: usize, draft_len: usize, next: i32)
    -> Result<(usize, i32), String>
{
    if accepted > draft_len {
        return Err(format!("verifier accepted {accepted} tokens from a {draft_len}-token draft"));
    }
    if next < 0 {
        return Err(format!("verifier returned invalid next token ID {next}"));
    }
    Ok((accepted, next))
}

#[cfg(test)]
mod speculative_cli_tests {
    use super::validate_verified_draft;

    #[test]
    fn accepts_valid_full_and_partial_draft_results() {
        assert_eq!(validate_verified_draft(0, 4, 9), Ok((0, 9)));
        assert_eq!(validate_verified_draft(2, 4, 9), Ok((2, 9)));
        assert_eq!(validate_verified_draft(4, 4, 9), Ok((4, 9)));
    }

    #[test]
    fn rejects_impossible_accept_count_and_negative_token() {
        assert!(validate_verified_draft(5, 4, 9).is_err());
        assert!(validate_verified_draft(0, 4, -1).is_err());
    }
}

#[cfg(test)]
mod utf8_stream_tests {
    use super::decode_utf8_piece;

    #[test]
    fn handles_multibyte_character_split_across_tokens() {
        let mut pending = Vec::new();
        assert_eq!(decode_utf8_piece(&mut pending, &[0xe2]), "");
        assert_eq!(decode_utf8_piece(&mut pending, &[0x82]), "");
        assert_eq!(decode_utf8_piece(&mut pending, &[0xac]), "€");
        assert!(pending.is_empty());
    }

    #[test]
    fn preserves_valid_prefix_while_buffering_incomplete_suffix() {
        let mut pending = Vec::new();
        assert_eq!(decode_utf8_piece(&mut pending, &[b'A', 0xc3]), "A");
        assert_eq!(decode_utf8_piece(&mut pending, &[0xa9]), "é");
    }
}

/// Truncate display text by Unicode scalar values, never by UTF-8 byte offset.
fn truncate_for_display(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", prefix)
    } else {
        prefix
    }
}

#[cfg(test)]
mod cli_tests {
    use super::{truncate_for_display, Cli, Commands};
    use clap::Parser;

    #[test]
    fn native_greedy_remains_explicit_opt_in() {
        let default = Cli::try_parse_from(["kiln", "run", "model.gguf"]).unwrap();
        assert!(matches!(default.command,
            Commands::Run { native_greedy: false, .. }));
        let selected = Cli::try_parse_from(["kiln", "run", "model.gguf", "--native-greedy"]).unwrap();
        assert!(matches!(selected.command,
            Commands::Run { native_greedy: true, .. }));
        let baseline_bench = Cli::try_parse_from(["kiln", "bench", "model.gguf"]).unwrap();
        assert!(matches!(baseline_bench.command,
            Commands::Bench { native_greedy: false, .. }));
        let tuned = Cli::try_parse_from(["kiln", "bench", "model.gguf", "--native-greedy"]).unwrap();
        assert!(matches!(tuned.command,
            Commands::Bench { native_greedy: true, .. }));
    }

    #[test]
    fn keep_context_is_explicit_opt_in() {
        let regular = Cli::try_parse_from(["kiln", "run", "model.gguf"]).unwrap();
        assert!(matches!(regular.command, Commands::Run { keep_context: false, .. }));
        let opted = Cli::try_parse_from([
            "kiln", "run", "model.gguf", "--keep-context", "--max-tokens", "32",
        ]).unwrap();
        assert!(matches!(opted.command, Commands::Run {
            keep_context: true, max_tokens: 32, ..
        }));
    }

    #[test]
    fn ngram_speculation_is_explicit_and_conflicts_with_plain_native_greedy_flag() {
        let selected = Cli::try_parse_from([
            "kiln", "run", "model.gguf", "--spec-type", "ngram-mod",
        ]).unwrap();
        assert!(matches!(selected.command, Commands::Run {
            spec_type: Some(ref kind), native_greedy: false, ..
        } if kind == "ngram-mod"));
        assert!(Cli::try_parse_from([
            "kiln", "run", "model.gguf", "--spec-type", "unknown",
        ]).is_err());
        assert!(Cli::try_parse_from([
            "kiln", "run", "model.gguf", "--spec-type", "ngram-mod", "--native-greedy",
        ]).is_err());
    }

    #[test]
    fn display_truncation_preserves_unicode_boundaries() {
        assert_eq!(truncate_for_display("é界x", 2), "é界...");
        assert_eq!(truncate_for_display("é界", 2), "é界");
        assert_eq!(truncate_for_display("abc", 0), "...");
        assert_eq!(truncate_for_display("", 3), "");
    }
}

/// Track exactly which tokens were executed into the native KV cache.
/// This state never guesses prefix equality from decoded text or retokenization.
/// The most recently emitted token may not have been forwarded yet.
#[derive(Debug, Default)]
struct InteractiveKvState {
    cached_tokens: usize,
    pending_token: Option<i32>,
}

impl InteractiveKvState {
    /// Return (must_reset, tokens_to_forward). A reset is required if
    /// continuing would exceed the configured context including generation.
    fn prepare(&mut self, turn_tokens: Vec<i32>, max_new: usize, ctx_limit: usize)
        -> Result<(bool, Vec<i32>), String>
    {
        if turn_tokens.is_empty() {
            return Err("empty turn tokenization".to_string());
        }
        let needed_fresh = turn_tokens.len().checked_add(max_new)
            .ok_or_else(|| "context length arithmetic overflow".to_string())?;
        if needed_fresh > ctx_limit {
            return Err(format!(
                "turn requires at least {needed_fresh} context tokens, but configured n_ctx is {ctx_limit}; shorten the prompt or lower --max-tokens"
            ));
        }
        let needed_continued = self.cached_tokens
            .checked_add(turn_tokens.len())
            .and_then(|n| n.checked_add(usize::from(self.pending_token.is_some())))
            .and_then(|n| n.checked_add(max_new))
            .unwrap_or(usize::MAX);
        if needed_continued > ctx_limit {
            self.cached_tokens = 0;
            self.pending_token = None;
            return Ok((true, turn_tokens));
        }
        let mut tail = Vec::with_capacity(turn_tokens.len() + usize::from(self.pending_token.is_some()));
        if let Some(pending) = self.pending_token.take() {
            tail.push(pending);
        }
        tail.extend(turn_tokens);
        Ok((false, tail))
    }

    fn forwarded(&mut self, count: usize) {
        self.cached_tokens += count;
    }

    fn emitted(&mut self, token: i32) {
        self.pending_token = Some(token);
    }

    /// EOS was sampled but never decoded; it must not enter next turn's KV.
    fn ended_on_eos(&mut self) {
        self.pending_token = None;
    }
}

#[cfg(test)]
mod interactive_kv_tests {
    use super::InteractiveKvState;

    #[test]
    fn appends_only_unexecuted_token_and_new_turn() {
        let mut cache = InteractiveKvState::default();
        let (reset, tail) = cache.prepare(vec![1, 2], 3, 20).unwrap();
        assert!(!reset);
        assert_eq!(tail, vec![1, 2]);
        cache.forwarded(tail.len());
        cache.emitted(7);
        let (reset, tail) = cache.prepare(vec![3, 4], 3, 20).unwrap();
        assert!(!reset);
        assert_eq!(tail, vec![7, 3, 4]);
        assert_eq!(cache.cached_tokens, 2);
    }

    #[test]
    fn resets_before_context_overflow_and_drops_pending() {
        let mut cache = InteractiveKvState { cached_tokens: 12, pending_token: Some(42) };
        let (reset, tail) = cache.prepare(vec![1, 2], 4, 16).unwrap();
        assert!(reset);
        assert_eq!(tail, vec![1, 2]);
        assert_eq!(cache.cached_tokens, 0);
        assert_eq!(cache.pending_token, None);
    }

    #[test]
    fn rejects_single_turn_too_large_without_mutating_state() {
        let mut cache = InteractiveKvState { cached_tokens: 2, pending_token: Some(5) };
        assert!(cache.prepare(vec![1, 2, 3], 6, 8).is_err());
        assert_eq!(cache.cached_tokens, 2);
        assert_eq!(cache.pending_token, Some(5));
    }

    #[test]
    fn eos_is_never_appended_to_next_turn() {
        let mut cache = InteractiveKvState::default();
        let (_, first) = cache.prepare(vec![10, 11], 3, 32).unwrap();
        cache.forwarded(first.len());
        // Simulate one emitted token that was decoded, then model EOS.
        cache.emitted(55);
        // The emitted token has now been evaluated (as in the decode loop).
        cache.pending_token = None;
        cache.forwarded(1);
        // EOS is sampled from that evaluation and is never forwarded.
        cache.ended_on_eos();
        let (reset, next) = cache.prepare(vec![20, 21], 3, 32).unwrap();
        assert!(!reset);
        assert_eq!(next, vec![20, 21]);
        assert_eq!(cache.cached_tokens, 3);
    }

    #[test]
    fn budget_excludes_eos_when_it_was_not_forwarded() {
        let mut cache = InteractiveKvState {
            cached_tokens: 5,
            pending_token: Some(99),
        };
        cache.ended_on_eos();
        let (reset, tail) = cache.prepare(vec![8, 9], 2, 9).unwrap();
        assert!(!reset);
        assert_eq!(tail, vec![8, 9]);
    }

    #[test]
    fn exact_fit_keeps_context() {
        let mut cache = InteractiveKvState { cached_tokens: 3, pending_token: Some(8) };
        let (reset, tail) = cache.prepare(vec![1, 2], 4, 10).unwrap();
        assert!(!reset);
        assert_eq!(tail, vec![8, 1, 2]);
    }
}

/// Run a real interactive text session using the runtime dispatcher.
fn run_interactive(model: &str, max_tokens: usize, keep_context: bool, native_greedy: bool, speculative: bool) -> Result<(), String> {
    use kiln_core::chat::{Sampler, SamplingStrategy};
    use kiln_runtime::{draft_ngram, LlamaCppBackend, NgramDraftConfig, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};
    use std::io::{self, Write};

    let profile = SystemProfile::load_or_probe();
    let mut dispatcher = RuntimeDispatcher::new();
    dispatcher.register(Box::new(LlamaCppBackend::new()));
    dispatcher.dispatch(RuntimeOp::Load {
        path: std::path::PathBuf::from(model),
        n_ctx: profile.n_ctx,
        n_batch: profile.n_batch,
        n_ubatch: profile.n_batch.min(512),
        n_threads: profile.n_threads,
    }).map_err(|e| format!("model load failed: {}", e))?;

    let eos = match dispatcher.dispatch(RuntimeOp::EosToken)
        .map_err(|e| format!("could not read EOS token: {}", e))? {
        RuntimeResult::Int(token) => token,
        _ => return Err("backend returned an unexpected EOS-token result".to_string()),
    };

    println!("KILN interactive session — model: {}", model);
    println!("Type /exit to quit. Profile: {:?}, threads={}, context={}",
        profile.tier, profile.n_threads, profile.n_ctx);
    if speculative {
        println!("Speculative decode: ngram-mod (experimental; target-verified)");
    }

    let stdin = io::stdin();
    let mut kv_state = InteractiveKvState::default();
    let mut speculative_history = Vec::<i32>::new();
    let draft_config = NgramDraftConfig::default();
    let fast_greedy = native_greedy || speculative;
    let mut first_turn = true;
    loop {
        print!("you> ");
        io::stdout().flush().map_err(|e| format!("prompt output failed: {}", e))?;
        let mut prompt = String::new();
        let bytes = stdin.read_line(&mut prompt).map_err(|e| format!("input failed: {}", e))?;
        if bytes == 0 {
            break;
        }
        let prompt = prompt.trim();
        if prompt.eq_ignore_ascii_case("/exit") || prompt.eq_ignore_ascii_case("/quit") {
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        // Keep the existing independent, raw-prompt behavior by default.
        // The opt-in mode builds a token stream incrementally and never
        // re-prefills a prefix that already exists in the native KV cache.
        let prompt_text = if keep_context {
            if first_turn {
                format!("User: {prompt}\nAssistant:")
            } else {
                format!("\nUser: {prompt}\nAssistant:")
            }
        } else {
            prompt.to_string()
        };
        let tokens = match dispatcher.dispatch(RuntimeOp::Tokenize { text: prompt_text })
            .map_err(|e| format!("tokenization failed: {}", e))? {
            RuntimeResult::Tokens(tokens) => tokens,
            _ => return Err("backend returned an unexpected tokenization result".to_string()),
        };
        if tokens.is_empty() {
            eprintln!("warning: prompt tokenized to no tokens");
            continue;
        }
        let previous_cached = kv_state.cached_tokens;
        let tokens = if keep_context {
            let (must_reset, tail) = kv_state.prepare(tokens, max_tokens, profile.n_ctx as usize)?;
            if must_reset {
                // After evicting the old dialogue, restart the prompt without
                // the continuation-only leading newline.
                let fresh = match dispatcher.dispatch(RuntimeOp::Tokenize {
                    text: format!("User: {prompt}\nAssistant:"),
                }).map_err(|e| format!("fresh turn tokenization failed: {e}"))? {
                    RuntimeResult::Tokens(tokens) => tokens,
                    _ => return Err("backend returned an unexpected tokenization result".to_string()),
                };
                let (_, fresh) = kv_state.prepare(fresh, max_tokens, profile.n_ctx as usize)?;
                dispatcher.dispatch(RuntimeOp::Reset)
                    .map_err(|e| format!("KV reset on context boundary failed: {e}"))?;
                speculative_history.clear();
                eprintln!("note: context limit approached; starting a fresh conversation window");
                fresh
            } else {
                tail
            }
        } else {
            dispatcher.dispatch(RuntimeOp::Reset)
                .map_err(|e| format!("session reset failed: {}", e))?;
            speculative_history.clear();
            tokens
        };
        let prefill_count = tokens.len();
        let prefill_tokens = if speculative { Some(tokens.clone()) } else { None };
        let prefill_start = std::time::Instant::now();
        let (mut logits, mut native_next) = if fast_greedy {
            match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens })
                .map_err(|e| format!("native greedy prompt evaluation failed: {e}"))? {
                RuntimeResult::Int(token) => (None, Some(token)),
                _ => return Err("backend returned unexpected native token".to_string()),
            }
        } else {
            match dispatcher.dispatch(RuntimeOp::Forward { tokens })
                .map_err(|e| format!("prompt evaluation failed: {e}"))? {
                RuntimeResult::Logits(logits) => (Some(logits), None),
                _ => return Err("backend returned an unexpected prompt result".to_string()),
            }
        };
        if keep_context {
            kv_state.forwarded(prefill_count);
            first_turn = false;
            tracing::debug!(
                prefill_tokens = prefill_count,
                cached_before_turn = previous_cached,
                cached_after_prefill = kv_state.cached_tokens,
                prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
                "append-only interactive prefill"
            );
        }
        if let Some(tokens) = prefill_tokens {
            speculative_history.extend(tokens);
        }

        print!("kiln> ");
        io::stdout().flush().map_err(|e| format!("response output failed: {}", e))?;
        let mut sampler = Sampler::new(SamplingStrategy::Greedy, 42);
        let mut pending_utf8 = Vec::new();
        let mut generated = 0usize;
        let mut proposed_draft_tokens = 0usize;
        let mut accepted_draft_tokens = 0usize;
        let generation_started = std::time::Instant::now();
        while generated < max_tokens {
            if speculative {
                let mut draft = draft_ngram(&speculative_history, draft_config, Some(eos));
                draft.truncate(max_tokens - generated);
                if !draft.is_empty() {
                    proposed_draft_tokens += draft.len();
                    let verification = match dispatcher.dispatch(RuntimeOp::VerifyGreedyDraft {
                        tokens: draft.clone(),
                    }).map_err(|e| format!("n-gram target verification failed: {e}"))? {
                        RuntimeResult::VerifiedGreedy { accepted, next } => (accepted, next),
                        _ => return Err("backend returned an unexpected n-gram verification result".to_string()),
                    };
                    let (accepted, corrected_next) = validate_verified_draft(
                        verification.0, draft.len(), verification.1,
                    )?;
                    accepted_draft_tokens += accepted;
                    // The verifier retains only this accepted prefix in KV and
                    // rolls back rejected/unverified draft positions.
                    speculative_history.extend_from_slice(&draft[..accepted]);
                    if keep_context {
                        kv_state.forwarded(accepted);
                    }
                    for &draft_token in &draft[..accepted] {
                        let piece = match dispatcher.dispatch(RuntimeOp::TokenToStr { token: draft_token })
                            .map_err(|e| format!("speculative token decoding failed: {e}"))? {
                            RuntimeResult::Bytes(bytes) => decode_utf8_piece(&mut pending_utf8, &bytes),
                            RuntimeResult::Str(text) => text,
                            _ => return Err("backend returned an unexpected speculative token-string result".to_string()),
                        };
                        print!("{}", piece);
                        io::stdout().flush().map_err(|e| format!("response output failed: {e}"))?;
                        generated += 1;
                    }
                    if generated >= max_tokens {
                        break;
                    }
                    native_next = Some(corrected_next);
                }
            }

            let token = if fast_greedy {
                native_next.ok_or_else(|| "no native next token".to_string())?
            } else {
                sampler.sample(logits.as_deref().ok_or("missing logits")?)
                    .ok_or_else(|| "sampler returned no token".to_string())? as i32
            };
            if token == eos {
                // EOS is not an output token and is not evaluated into the
                // native KV cache. Forwarding it at the next turn would
                // change the continuation compared with the executed prefix.
                if keep_context { kv_state.ended_on_eos(); }
                break;
            }
            if keep_context { kv_state.emitted(token); }
            let piece = match dispatcher.dispatch(RuntimeOp::TokenToStr { token })
                .map_err(|e| format!("token decoding failed: {}", e))? {
                RuntimeResult::Bytes(bytes) => decode_utf8_piece(&mut pending_utf8, &bytes),
                RuntimeResult::Str(text) => text,
                _ => return Err("backend returned an unexpected token-string result".to_string()),
            };
            print!("{}", piece);
            io::stdout().flush().map_err(|e| format!("response output failed: {}", e))?;
            generated += 1;
            if generated < max_tokens {
                if fast_greedy {
                    native_next = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: vec![token] })
                        .map_err(|e| format!("native greedy decode failed: {e}"))? {
                        RuntimeResult::Int(next) => Some(next),
                        _ => return Err("backend returned unexpected native token".to_string()),
                    };
                } else {
                    logits = match dispatcher.dispatch(RuntimeOp::Forward { tokens: vec![token] })
                        .map_err(|e| format!("decode step failed: {e}"))? {
                        RuntimeResult::Logits(logits) => Some(logits),
                        _ => return Err("backend returned an unexpected decode result".to_string()),
                    };
                }
                if speculative {
                    speculative_history.push(token);
                }
                if keep_context {
                    kv_state.pending_token = None;
                    kv_state.forwarded(1);
                }
            }
        }
        if !pending_utf8.is_empty() {
            print!("{}", String::from_utf8_lossy(&pending_utf8));
        }
        println!();
        if speculative {
            let elapsed = generation_started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            eprintln!("[ngram-mod: {} proposed, {} accepted ({:.1}%), {:.2} generated tok/s]",
                proposed_draft_tokens,
                accepted_draft_tokens,
                if proposed_draft_tokens == 0 { 0.0 }
                    else { accepted_draft_tokens as f64 * 100.0 / proposed_draft_tokens as f64 },
                generated as f64 / elapsed);
        }
    }
    dispatcher.dispatch(RuntimeOp::Unload)
        .map_err(|e| format!("model unload failed: {}", e))?;
    Ok(())
}

/// Execute one benchmark sample through the same runtime dispatcher used by chat.
fn bench_once(
    dispatcher: &mut kiln_runtime::RuntimeDispatcher,
    prompt: &str,
    max_tokens: usize,
    eos_token: i32,
    native_greedy: bool,
) -> Result<serde_json::Value, String> {
    use kiln_core::chat::{Sampler, SamplingStrategy};
    use kiln_runtime::{RuntimeOp, RuntimeResult};

    dispatcher.dispatch(RuntimeOp::Reset).map_err(|e| format!("reset failed: {}", e))?;
    let tokenization_start = std::time::Instant::now();
    let prompt_tokens = match dispatcher.dispatch(RuntimeOp::Tokenize { text: prompt.to_string() })
        .map_err(|e| format!("tokenization failed: {}", e))? {
        RuntimeResult::Tokens(tokens) => tokens,
        _ => return Err("backend returned an unexpected tokenization result".to_string()),
    };
    let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_string());
    }

    let prefill_start = std::time::Instant::now();
    let mut next_logits: Option<Vec<f32>> = None;
    let mut native_next: Option<i32> = None;
    if native_greedy {
        native_next = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: prompt_tokens.clone() })
            .map_err(|e| format!("native greedy prefill failed: {e}"))? {
            RuntimeResult::Int(token) => Some(token),
            _ => return Err("unexpected native greedy prefill result".into()),
        };
    } else {
        next_logits = match dispatcher.dispatch(RuntimeOp::Forward { tokens: prompt_tokens.clone() })
            .map_err(|e| format!("prompt forward pass failed: {e}"))? {
            RuntimeResult::Logits(logits) => Some(logits),
            _ => return Err("unexpected prefill result".into()),
        };
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut sampler = Sampler::new(SamplingStrategy::Greedy, 42);
    let mut generated_tokens = 0usize;
    // Stable, allocation-free digest of the exact generated token sequence;
    // compare it between baseline and native greedy runs on the same model.
    let mut token_fingerprint_fnv1a64 = 0xcbf29ce484222325u64;
    let mut eos_encountered = false;
    let mut sampling_ms = 0.0_f64;
    let mut token_conversion_ms = 0.0_f64;
    let mut decode_forward_ms = 0.0_f64;
    let mut decode_forward_steps = 0usize;
    let decode_start = std::time::Instant::now();
    for step in 0..max_tokens {
        let sample_start = std::time::Instant::now();
        let token = if native_greedy {
            native_next.ok_or_else(|| "no native greedy token available".to_string())?
        } else {
            sampler.sample(next_logits.as_deref().ok_or("missing logits")?)
                .ok_or_else(|| "sampler returned no token".to_string())? as i32
        };
        sampling_ms += sample_start.elapsed().as_secs_f64() * 1000.0;
        if token == eos_token {
            eos_encountered = true;
            break;
        }
        for byte in token.to_le_bytes() {
            token_fingerprint_fnv1a64 ^= u64::from(byte);
            token_fingerprint_fnv1a64 = token_fingerprint_fnv1a64.wrapping_mul(0x100000001b3);
        }

        let conversion_start = std::time::Instant::now();
        match dispatcher.dispatch(RuntimeOp::TokenToStr { token })
            .map_err(|e| format!("token-to-string failed: {}", e))? {
            RuntimeResult::Bytes(_) | RuntimeResult::Str(_) => {}
            _ => return Err("backend returned an unexpected token conversion result".to_string()),
        }
        token_conversion_ms += conversion_start.elapsed().as_secs_f64() * 1000.0;
        generated_tokens += 1;
        // The final requested output token does not need a follow-up forward
        // pass. Counting that extra pass would distort the decode benchmark.
        if step + 1 < max_tokens {
            let forward_start = std::time::Instant::now();
            if native_greedy {
                native_next = match dispatcher.dispatch(RuntimeOp::ForwardGreedy { tokens: vec![token] })
                    .map_err(|e| format!("native decode forward failed: {e}"))? {
                    RuntimeResult::Int(next) => Some(next),
                    _ => return Err("backend returned an unexpected native decode result".into()),
                };
            } else {
                next_logits = match dispatcher.dispatch(RuntimeOp::Forward { tokens: vec![token] })
                    .map_err(|e| format!("decode forward pass failed: {}", e))? {
                    RuntimeResult::Logits(logits) => Some(logits),
                    _ => return Err("backend returned an unexpected decode result".to_string()),
                };
            }
            decode_forward_ms += forward_start.elapsed().as_secs_f64() * 1000.0;
            decode_forward_steps += 1;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let decode_ms = decode_seconds * 1000.0;
    // Legacy metric counts output samples, including the first (prefill-derived)
    // token. Do not compare it with llama.cpp's token-decode throughput.
    let decode_tokens_per_second = if decode_seconds > 0.0 {
        generated_tokens as f64 / decode_seconds
    } else { 0.0 };
    let decode_forward_tokens_per_second = if decode_forward_ms > 0.0 {
        decode_forward_steps as f64 * 1000.0 / decode_forward_ms
    } else { 0.0 };
    let full_response_ms = prefill_ms + decode_ms;
    let response_tokens_per_second_including_prefill = if full_response_ms > 0.0 {
        generated_tokens as f64 * 1000.0 / full_response_ms
    } else { 0.0 };

    Ok(serde_json::json!({
        "prompt_tokens": prompt_tokens.len(),
        "sampling_path": if native_greedy { "native_greedy" } else { "rust_logits_greedy" },
        "generated_tokens": generated_tokens,
        "token_fingerprint_fnv1a64": format!("{token_fingerprint_fnv1a64:016x}"),
        "eos_encountered": eos_encountered,
        "tokenization_ms": tokenization_ms,
        "prefill_ms": prefill_ms,
        "sampling_ms": sampling_ms,
        "token_conversion_ms": token_conversion_ms,
        "decode_forward_ms": decode_forward_ms,
        "decode_forward_steps": decode_forward_steps,
        "decode_forward_tokens_per_second": decode_forward_tokens_per_second,
        "response_tokens_per_second_including_prefill": response_tokens_per_second_including_prefill,
        "decode_ms": decode_ms,
        "decode_tokens_per_second": decode_tokens_per_second
    }))
}

fn not_implemented(cmd: &str, arg: &str) -> ! {
    if arg.is_empty() {
        eprintln!("kiln {}: not yet implemented in Phase 0B", cmd);
    } else {
        eprintln!("kiln {} {}: not yet implemented in Phase 0B", cmd, arg);
    }
    std::process::exit(2);
}
