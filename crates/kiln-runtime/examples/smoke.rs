//! Smoke test. Load a model via llama.cpp, tokenize, run forward
//! N times, report per-forward time. Matches llama-bench's methodology
//! so the comparison is apples to apples.

use kiln_runtime::LlamaContext;
use std::path::Path;

fn main() {
    let model = std::env::args().nth(1)
        .unwrap_or_else(|| "models/tiny/qwen25-1.5b.gguf".to_string());
    let prompt = std::env::args().nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let iters: usize = std::env::args().nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("model:  {}", model);
    println!("prompt: {:?}", prompt);
    println!("iters:  {}", iters);

    let mut ctx = match LlamaContext::load(Path::new(&model)) {
        Ok(c) => c,
        Err(e) => { eprintln!("load failed: {}", e); std::process::exit(1); }
    };
    println!("n_vocab = {}", ctx.n_vocab());

    let tokens = match ctx.tokenize(&prompt) {
        Ok(t) => t,
        Err(e) => { eprintln!("tokenize failed: {}", e); std::process::exit(1); }
    };
    println!("tokens: {:?}", tokens);

    // Warmup. First run allocates the graph.
    let _ = ctx.forward(&tokens);

    // Reset KV cache by constructing a fresh context for every timed
    // iteration, but keep the model loaded. The shim does not expose a
    // cache reset yet, so for prefill measurement we re-load and
    // re-run. Report both the per-iteration wall and the pure forward
    // time from inside the context.
    let mut total_fwd = 0.0f64;
    for i in 0..iters {
        let mut c = match LlamaContext::load(Path::new(&model)) {
            Ok(c) => c,
            Err(e) => { eprintln!("load failed iter {}: {}", i, e); std::process::exit(1); }
        };
        let toks = c.tokenize(&prompt).unwrap();
        let t = std::time::Instant::now();
        let logits = c.forward(&toks).unwrap();
        let fwd = t.elapsed().as_secs_f64();
        total_fwd += fwd;
        if i == 0 {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            println!("top 5:");
            for k in 0..5.min(idx.len()) {
                let ii = idx[k];
                let s = c.token_to_str(ii as i32);
                println!("  {:>7}  {:>10.4}  {:?}", ii, logits[ii], s);
            }
        }
    }
    let avg = total_fwd / iters as f64;
    println!("forward avg over {} iters: {:.4}s ({:.2} tok/s for {} tokens)",
             iters, avg, tokens.len() as f64 / avg, tokens.len());
}
