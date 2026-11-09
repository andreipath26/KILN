//! Deterministic, model-agnostic cross-entropy evidence for quantization parity.
//! This is a bounded, reset-window next-token scoring protocol, not a
//! fabricated WikiText-2 score. FP16/reference and candidate must use the
//! identical text, tokenizer IDs and window settings for a valid comparison.
use std::path::Path;
use std::time::Instant;
use kiln_runtime::{LlamaContext, SystemProfile};
use serde_json::json;
use sha2::{Digest, Sha256};

const MAX_CORPUS_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVAL_TOKENS: usize = 65_536;

fn log_probability(logits: &[f32], target: i32) -> Result<f64, String> {
    let index = usize::try_from(target).map_err(|_| "negative target token id")?;
    let wanted = *logits.get(index).ok_or("target token id exceeds vocabulary")? as f64;
    // A -infinity logit is a legitimate zero-probability masked vocabulary
    // entry. Reject NaN/+infinity everywhere and a masked target, but do not
    // declare normal model logits invalid merely because EOG IDs are masked.
    if logits.is_empty() || !wanted.is_finite() ||
        logits.iter().any(|value| value.is_nan() || *value == f32::INFINITY) {
        return Err("invalid target or NaN/+infinity vocabulary logit".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !maximum.is_finite() {
        return Err("no finite logits remain after vocabulary masking".into());
    }
    let normalizer: f64 = logits.iter()
        .map(|value| ((*value as f64) - maximum).exp()).sum();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err("invalid softmax normalizer".into());
    }
    let log_prob = wanted - maximum - normalizer.ln();
    if !log_prob.is_finite() || log_prob > 1e-8 {
        return Err("invalid token log probability".into());
    }
    Ok(log_prob)
}

fn token_fingerprint(tokens: &[i32]) -> String {
    let mut state = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            state ^= u64::from(byte);
            state = state.wrapping_mul(0x100000001b3);
        }
    }
    format!("{state:016x}")
}

pub fn run(
    model: &str,
    text_file: &Path,
    max_tokens: usize,
    window_tokens: usize,
    threads: Option<u32>,
    lazy_moe_experts: bool,
    json_output: bool,
) -> Result<(), String> {
    if max_tokens < 2 || max_tokens > MAX_EVAL_TOKENS {
        return Err(format!("--max-tokens must be in 2..={MAX_EVAL_TOKENS}"));
    }
    if window_tokens < 2 || window_tokens > 8192 {
        return Err("--window-tokens must be in 2..=8192".into());
    }
    let size = std::fs::metadata(text_file)
        .map_err(|error| format!("corpus metadata: {error}"))?.len();
    if size == 0 || size > MAX_CORPUS_BYTES {
        return Err(format!("corpus must have 1..={MAX_CORPUS_BYTES} bytes"));
    }
    let corpus = std::fs::read_to_string(text_file)
        .map_err(|error| format!("corpus is not UTF-8 text: {error}"))?;
    let path = crate::model_library::resolve_model_identifier(model)?;
    let mut profile = SystemProfile::load_or_probe();
    let cpu_threads = threads.unwrap_or(profile.n_threads);
    if cpu_threads == 0 || cpu_threads > 1024 {
        return Err("--threads must be 1..=1024".into());
    }
    let context_size = u32::try_from(window_tokens)
        .map_err(|_| "invalid context window")?;
    profile.n_ctx = context_size;
    let parameters = (
        i32::try_from(context_size).map_err(|_| "invalid context size")?,
        i32::try_from(profile.n_batch.min(512)).map_err(|_| "invalid batch size")?,
        i32::try_from(profile.n_batch.min(512)).map_err(|_| "invalid ubatch")?,
        i32::try_from(cpu_threads).map_err(|_| "invalid thread count")?,
    );
    let mut backend = LlamaContext::load_with_ubatch_and_lazy_moe(
        &path, Some(parameters), lazy_moe_experts
    ).map_err(|error| format!("model load: {error}"))?;
    let mut tokens = backend.tokenize(&corpus).map_err(|e| e.to_string())?;
    if tokens.len() < 2 {
        return Err("corpus tokenized to fewer than two tokens".into());
    }
    tokens.truncate(max_tokens);
    let fingerprint = token_fingerprint(&tokens);
    // Record cryptographic corpus token identity, not only a compact FNV
    // diagnostic, before allowing FP16-vs-candidate likelihood comparisons.
    let mut digest = Sha256::new();
    for token in &tokens {
        digest.update(token.to_le_bytes());
    }
    let token_sha256 = format!("{:x}", digest.finalize());
    let mut total_nll = 0.0f64;
    let mut scored = 0usize;
    let mut windows = 0usize;
    let time_start = Instant::now();
    let mut beginning = 0usize;
    while beginning + 1 < tokens.len() {
        let end = (beginning + window_tokens).min(tokens.len());
        backend.reset().map_err(|e| format!("context reset failed: {e}"))?;
        windows += 1;
        for cursor in beginning..end - 1 {
            let logits = backend.forward(&[tokens[cursor]])
                .map_err(|e| format!("eval token {cursor}: {e}"))?;
            let log_prob = log_probability(&logits, tokens[cursor + 1])?;
            total_nll -= log_prob;
            scored += 1;
        }
        beginning = end - 1;
    }
    if scored != tokens.len() - 1 {
        return Err("quality evaluation silently skipped or duplicated target positions".into());
    }
    let average_nll = total_nll / scored as f64;
    let perplexity = average_nll.exp();
    if !perplexity.is_finite() {
        return Err("non-finite perplexity: quality evidence invalid".into());
    }
    let report = json!({
        "protocol": "kiln-reset-window-next-token-nll-v1",
        "corpus_path": std::fs::canonicalize(text_file)
            .unwrap_or_else(|_| text_file.to_path_buf()),
        "model_path": std::fs::canonicalize(&path).unwrap_or(path),
        "token_fingerprint_fnv1a64": fingerprint,
        "token_ids_sha256_le_i32": token_sha256,
        "evaluated_token_count": tokens.len(),
        "scored_target_tokens": scored,
        "window_tokens": window_tokens,
        "context_resets": windows,
        "log_likelihood_nats": -total_nll,
        "mean_nll_nats": average_nll,
        "perplexity": perplexity,
        "elapsed_seconds": time_start.elapsed().as_secs_f64(),
        "cpu_threads": cpu_threads,
        "lazy_moe_experts": lazy_moe_experts,
        "deployment_accepted": false,
        "note": "Same tokenization, corpus and scoring protocol required for FP16 vs candidate comparison. This is a reset-window perplexity, not the official WikiText-2 full-benchmark score."
    });
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        println!("KILN exact next-token quality evidence (bounded reset-window protocol)");
        println!("  model: {}", model);
        println!("  token fingerprint: {}", fingerprint);
        println!("  targets: {scored} in {windows} windows");
        println!("  mean negative log-likelihood: {average_nll:.6} nats");
        println!("  perplexity: {perplexity:.6}");
        println!("  Not an official WikiText-2 benchmark or deployment acceptance.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn log_sum_exp_is_stable_and_normalized() {
        let p = log_probability(&[0.,0.,0.],1).unwrap();
        assert!((p + 3f64.ln()).abs() < 1e-12);
        let extreme = log_probability(&[-10000., 10000.],1).unwrap();
        assert!(extreme.abs() < 1e-12);
        assert!(log_probability(&[1.0, f32::NAN],0).is_err());
        assert!(log_probability(&[1.0],3).is_err());
        assert!(log_probability(&[0.0, f32::NEG_INFINITY], 0).unwrap().abs() < 1e-12);
        assert!(log_probability(&[0.0, f32::NEG_INFINITY], 1).is_err());
        assert!(log_probability(&[f32::NEG_INFINITY], 0).is_err());
    }
    #[test]
    fn same_token_sequence_has_stable_cross_platform_fingerprint() {
        assert_eq!(token_fingerprint(&[1,2,3]), token_fingerprint(&[1,2,3]));
        assert_ne!(token_fingerprint(&[1,2,3]), token_fingerprint(&[1,3,2]));
    }
}
