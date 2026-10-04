//! Smoke test: load a model via llama.cpp, tokenize, forward, print
//! the top 3 logits. This is the first time KILN uses llama.cpp's
//! actual inference.

use kiln_runtime::LlamaContext;
use std::path::Path;

fn main() {
    let model = std::env::args().nth(1)
        .unwrap_or_else(|| "models/tiny/qwen25-1.5b.gguf".to_string());
    let prompt = std::env::args().nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());

    println!("model:  {}", model);
    println!("prompt: {:?}", prompt);

    let t0 = std::time::Instant::now();
    let mut ctx = match LlamaContext::load(Path::new(&model)) {
        Ok(c) => c,
        Err(e) => { eprintln!("load failed: {}", e); std::process::exit(1); }
    };
    println!("loaded in {:.2}s, n_vocab = {}", t0.elapsed().as_secs_f64(), ctx.n_vocab());

    let tokens = match ctx.tokenize(&prompt) {
        Ok(t) => t,
        Err(e) => { eprintln!("tokenize failed: {}", e); std::process::exit(1); }
    };
    println!("tokens: {:?}", tokens);

    let t1 = std::time::Instant::now();
    let logits = match ctx.forward(&tokens) {
        Ok(l) => l,
        Err(e) => { eprintln!("forward failed: {}", e); std::process::exit(1); }
    };
    println!("forward: {:.4}s", t1.elapsed().as_secs_f64());

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    println!("top 5:");
    for k in 0..5.min(idx.len()) {
        let i = idx[k];
        let s = ctx.token_to_str(i as i32);
        println!("  {:>7}  {:>10.4}  {:?}", i, logits[i], s);
    }
}
