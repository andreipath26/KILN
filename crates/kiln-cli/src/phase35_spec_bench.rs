//! Phase 3.5 draftless speculative decoding A/B with exact greedy parity.
//! This is real batched target verification, not a token-sequence simulator.
use kiln_runtime::{
    draft_ngram, LlamaCppBackend, NgramDraftConfig, RuntimeDispatcher, RuntimeOp,
    RuntimeResult, SystemProfile,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, serde::Serialize)]
struct Sample {
    generated: usize,
    eos: bool,
    decode_ms: f64,
    tokens_per_second: f64,
    target_forward_passes: usize,
    verification_passes: usize,
    proposed_draft_tokens: usize,
    accepted_draft_tokens: usize,
    token_fingerprint_fnv1a64: String,
    // Token IDs allow the A/B to reject rare digest collisions.
    tokens: Vec<i32>,
}

fn dispatch_token(d: &mut RuntimeDispatcher, op: RuntimeOp) -> Result<i32, String> {
    match d.dispatch(op).map_err(|e| e.to_string())? {
        RuntimeResult::Int(token) => Ok(token),
        _ => Err("expected native greedy token".into()),
    }
}

fn run_sample(
    d: &mut RuntimeDispatcher, prompt_ids: &[i32], eos: i32,
    max_tokens: usize, cfg: Option<NgramDraftConfig>,
) -> Result<Sample, String> {
    d.dispatch(RuntimeOp::Reset).map_err(|e| format!("reset: {e}"))?;
    let mut next = dispatch_token(
        d, RuntimeOp::ForwardGreedy { tokens: prompt_ids.to_vec() },
    )?;
    let mut history = prompt_ids.to_vec();
    let mut output = Vec::with_capacity(max_tokens);
    let mut forwards = 0usize;
    let mut verifies = 0usize;
    let mut proposed = 0usize;
    let mut accepted_total = 0usize;
    let mut eos_hit = false;
    let started = Instant::now();
    while output.len() < max_tokens {
        if let Some(config) = cfg {
            let remaining = max_tokens - output.len();
            let mut draft = draft_ngram(&history, config, (eos >= 0).then_some(eos));
            draft.truncate(remaining.min(config.max_draft));
            if !draft.is_empty() {
                proposed += draft.len();
                verifies += 1;
                let result = d.dispatch(RuntimeOp::VerifyGreedyDraft { tokens: draft.clone() })
                    .map_err(|e| format!("native verification: {e}"))?;
                let (accepted, corrected) = match result {
                    RuntimeResult::VerifiedGreedy { accepted, next } => (accepted, next),
                    _ => return Err("unexpected verification result".into()),
                };
                if accepted > draft.len() || corrected < 0 {
                    return Err("invalid target verification result".into());
                }
                accepted_total += accepted;
                history.extend_from_slice(&draft[..accepted]);
                output.extend_from_slice(&draft[..accepted]);
                next = corrected;
                if output.len() == max_tokens { break; }
            }
        }
        if next == eos {
            eos_hit = true;
            break;
        }
        if next < 0 {
            return Err("negative target token ID".into());
        }
        output.push(next);
        if output.len() == max_tokens { break; }
        let consumed = next;
        next = dispatch_token(d, RuntimeOp::ForwardGreedy { tokens: vec![consumed] })?;
        history.push(consumed);
        forwards += 1;
    }
    let secs = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    let mut hash = 0xcbf29ce484222325u64;
    for token in &output {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(Sample {
        generated: output.len(), eos: eos_hit, decode_ms: secs * 1000.0,
        tokens_per_second: output.len() as f64 / secs,
        target_forward_passes: forwards, verification_passes: verifies,
        proposed_draft_tokens: proposed, accepted_draft_tokens: accepted_total,
        token_fingerprint_fnv1a64: format!("{hash:016x}"), tokens: output,
    })
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    if xs.len() % 2 == 0 {
        (xs[xs.len()/2-1] + xs[xs.len()/2]) * 0.5
    } else {
        xs[xs.len()/2]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    model: String, prompt: String, tokens: usize, runs: usize, warmup: usize,
    threads: Option<u32>, batch: Option<u32>, ubatch: Option<u32>,
    ngram: usize, draft_max: usize, json: bool,
) -> Result<(), String> {
    if tokens == 0 || runs == 0 || runs > 100 || warmup > 100 {
        return Err("--tokens and --runs must be positive; runs/warmup <= 100".into());
    }
    let cfg = NgramDraftConfig { ngram, max_draft: draft_max, search_window: 8192 }
        .validate().map_err(str::to_string)?;
    let mut profile = SystemProfile::load_or_probe();
    if let Some(v) = threads { profile.n_threads = v; }
    if let Some(v) = batch { profile.n_batch = v; }
    let microbatch = ubatch.unwrap_or(profile.n_batch.min(512));
    if profile.n_threads == 0 || profile.n_batch == 0 || microbatch == 0
        || microbatch > profile.n_batch {
        return Err("threads, batch and ubatch must be positive; ubatch <= batch".into());
    }
    // Keep the native verification batch bounded to its logical n_batch.
    if cfg.max_draft > profile.n_batch as usize {
        return Err("draft_max exceeds the configured logical batch".into());
    }

    let mut d = RuntimeDispatcher::new();
    d.register(Box::new(LlamaCppBackend::new()));
    d.dispatch(RuntimeOp::Load {
        path: PathBuf::from(&model), n_ctx: profile.n_ctx,
        n_batch: profile.n_batch, n_ubatch: microbatch, n_threads: profile.n_threads,
    }).map_err(|e| format!("load: {e}"))?;
    let prompt_ids = match d.dispatch(RuntimeOp::Tokenize { text: prompt.clone() })
        .map_err(|e| e.to_string())? {
        RuntimeResult::Tokens(ids) if !ids.is_empty() => ids,
        _ => return Err("prompt tokenization produced no IDs".into()),
    };
    // Reserve space for the maximum requested outputs AND their verification
    // batches. No context-shifting is attempted in this acceptance harness.
    if prompt_ids.len().checked_add(tokens)
        .filter(|total| *total <= profile.n_ctx as usize).is_none() {
        return Err("prompt + output tokens exceed configured n_ctx".into());
    }
    let eos = dispatch_token(&mut d, RuntimeOp::EosToken)?;
    for _ in 0..warmup {
        let baseline = run_sample(&mut d, &prompt_ids, eos, tokens, None)?;
        let speculative = run_sample(&mut d, &prompt_ids, eos, tokens, Some(cfg))?;
        if baseline.tokens != speculative.tokens {
            return Err("speculation changed greedy output during warmup".into());
        }
    }
    let mut records = Vec::with_capacity(runs);
    let mut base_rates = Vec::new();
    let mut spec_rates = Vec::new();
    for i in 0..runs {
        let (baseline, speculative) = if i % 2 == 0 {
            (run_sample(&mut d, &prompt_ids, eos, tokens, None)?,
             run_sample(&mut d, &prompt_ids, eos, tokens, Some(cfg))?)
        } else {
            let speculative = run_sample(&mut d, &prompt_ids, eos, tokens, Some(cfg))?;
            let baseline = run_sample(&mut d, &prompt_ids, eos, tokens, None)?;
            (baseline, speculative)
        };
        if baseline.tokens != speculative.tokens || baseline.eos != speculative.eos {
            return Err(format!("speculation changed greedy token sequence on run {}", i + 1));
        }
        base_rates.push(baseline.tokens_per_second);
        spec_rates.push(speculative.tokens_per_second);
        records.push(serde_json::json!({ "baseline": baseline, "speculative": speculative }));
    }
    let base_med = median(base_rates);
    let spec_med = median(spec_rates);
    let result = serde_json::json!({
        "phase": "3.5",
        "method": "draftless_ngram_target_greedy_batched_verification",
        "model": model, "prompt": prompt, "prompt_tokens": prompt_ids.len(),
        "max_generated_tokens": tokens, "runs": runs, "warmup": warmup,
        "threads": profile.n_threads, "batch": profile.n_batch,
        "ubatch": microbatch, "ngram": cfg.ngram, "draft_max": cfg.max_draft,
        "baseline_median_tokens_per_second": base_med,
        "speculative_median_tokens_per_second": spec_med,
        "relative_rate": if base_med > 0.0 { spec_med / base_med } else { 0.0 },
        "all_output_sequences_equal": true,
        "note": "Target-model greedy parity checked on exact token IDs. This is not an acceptance test until run with pinned GGUF on Dell. Decode time excludes prompt prefill/model loading. No unverified tokens are emitted.",
        "samples": records,
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?);
    } else {
        println!("KILN Phase 3.5 draftless speculation A/B");
        println!("  Greedy output parity: PASS for {runs} runs");
        println!("  Baseline median: {base_med:.2} tokens/s");
        println!("  Speculative median: {spec_med:.2} tokens/s");
        println!("  Relative rate: {:.3}", result["relative_rate"].as_f64().unwrap_or(0.0));
        println!("  Dell acceptance still requires pinned real-model measurements.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Pure synthetic backend for exercising target verification and KV
    /// correction paths without downloading a model in CI.
    struct DeterministicTarget { kv: Vec<i32> }
    impl DeterministicTarget {
        fn next(&self) -> i32 {
            if self.kv.len() % 7 == 0 { 3 }
            else if self.kv.last() == Some(&1) { 2 } else { 1 }
        }
    }
    impl kiln_runtime::RuntimeBackend for DeterministicTarget {
        fn name(&self) -> &str { "deterministic-spec-target" }
        fn supports(&self, _op: &RuntimeOp) -> bool { true }
        fn estimate(&self, _op: &RuntimeOp) -> Option<u64> { Some(0) }
        fn execute(&mut self, op: &RuntimeOp)
            -> Result<RuntimeResult, kiln_runtime::RuntimeError>
        {
            match op {
                RuntimeOp::Reset => { self.kv.clear(); Ok(RuntimeResult::Unit) }
                RuntimeOp::ForwardGreedy { tokens } => {
                    self.kv.extend_from_slice(tokens);
                    Ok(RuntimeResult::Int(self.next()))
                }
                RuntimeOp::VerifyGreedyDraft { tokens } => {
                    let mut accepted=0;
                    for &token in tokens {
                        let correct=self.next();
                        if token != correct {
                            return Ok(RuntimeResult::VerifiedGreedy {
                                accepted, next: correct,
                            });
                        }
                        self.kv.push(token);
                        accepted+=1;
                    }
                    Ok(RuntimeResult::VerifiedGreedy {
                        accepted, next:self.next(),
                    })
                }
                _ => Err(kiln_runtime::RuntimeError::WrongReturn),
            }
        }
    }

    #[test]
    fn batched_speculation_matches_baseline_on_accept_and_reject() {
        let mut d=RuntimeDispatcher::new();
        d.register(Box::new(DeterministicTarget{kv:Vec::new()}));
        let prefix=[1,2,1,2,1,2];
        let baseline=run_sample(&mut d,&prefix,-1,40,None).unwrap();
        let spec=run_sample(&mut d,&prefix,-1,40,Some(
            NgramDraftConfig{ngram:2,search_window:128,max_draft:6}
        )).unwrap();
        assert_eq!(baseline.tokens,spec.tokens);
        assert_eq!(baseline.token_fingerprint_fnv1a64,spec.token_fingerprint_fnv1a64);
        assert_eq!(baseline.generated,40);
        assert!(spec.verification_passes>0);
        assert!(spec.accepted_draft_tokens>0);
        assert!(spec.proposed_draft_tokens>spec.accepted_draft_tokens,
            "forced target branch at position 7 must reject some predictions");
    }

    #[test]
    fn median_handles_even_and_odd_trials() {
        assert_eq!(median(vec![2.0, 1.0, 3.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), 2.5);
    }
    #[test]
    fn invalid_draft_parameters_rejected_before_loading() {
        assert!(run("missing".into(), "hi".into(), 10, 2, 0,
            Some(2), Some(512), Some(128), 0, 8, true).is_err());
    }
}
