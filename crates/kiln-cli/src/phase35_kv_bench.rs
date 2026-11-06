//! Controlled Phase 3.5 full re-prefill vs append-only KV benchmark.
//! The two candidates execute identical token sequences through RuntimeDispatcher.
use std::path::PathBuf;
use std::time::Instant;
use kiln_runtime::{LlamaCppBackend, RuntimeDispatcher, RuntimeOp, RuntimeResult, SystemProfile};

fn forward(dispatcher: &mut RuntimeDispatcher, tokens: Vec<i32>) -> Result<Vec<f32>, String> {
    match dispatcher.dispatch(RuntimeOp::Forward { tokens })
        .map_err(|e| format!("forward: {e}"))? {
        RuntimeResult::Logits(xs) => Ok(xs),
        _ => Err("backend returned unexpected forward result".into()),
    }
}

fn tokenize(dispatcher: &mut RuntimeDispatcher, text: &str) -> Result<Vec<i32>, String> {
    match dispatcher.dispatch(RuntimeOp::Tokenize { text: text.to_owned() })
        .map_err(|e| format!("tokenization: {e}"))? {
        RuntimeResult::Tokens(xs) if !xs.is_empty() => Ok(xs),
        _ => Err("backend returned empty/unexpected tokens".into()),
    }
}

fn reset(dispatcher: &mut RuntimeDispatcher) -> Result<(), String> {
    dispatcher.dispatch(RuntimeOp::Reset).map_err(|e| format!("reset: {e}"))?;
    Ok(())
}

fn top_token(xs: &[f32]) -> Option<usize> {
    if xs.is_empty() || xs.iter().any(|v| !v.is_finite()) { return None; }
    xs.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|x| x.0)
}

fn evaluate(
    dispatcher: &mut RuntimeDispatcher,
    first: &[i32],
    reply: &[i32],
    second: &[i32],
    cached: bool,
) -> Result<(f64, Vec<f32>), String> {
    reset(dispatcher)?;
    if cached {
        forward(dispatcher, first.to_vec())?;
        forward(dispatcher, reply.to_vec())?;
        let started = Instant::now();
        let logits = forward(dispatcher, second.to_vec())?;
        Ok((started.elapsed().as_secs_f64() * 1000.0, logits))
    } else {
        let mut all = Vec::with_capacity(first.len() + reply.len() + second.len());
        all.extend_from_slice(first);
        all.extend_from_slice(reply);
        all.extend_from_slice(second);
        let started = Instant::now();
        let logits = forward(dispatcher, all)?;
        Ok((started.elapsed().as_secs_f64() * 1000.0, logits))
    }
}

fn summary(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    let median = if values.len() % 2 == 1 { values[values.len()/2] }
        else { (values[values.len()/2-1] + values[values.len()/2]) / 2.0 };
    serde_json::json!({"median": median, "min": values[0], "max": values[values.len()-1]})
}

pub fn run(
    model: String,
    first_prompt: String,
    second_prompt: String,
    assistant_fragment: String,
    runs: usize,
    threads: Option<u32>,
    batch: Option<u32>,
    ubatch: Option<u32>,
    json: bool,
) -> Result<(), String> {
    if runs == 0 || runs > 100 {
        return Err("--runs must be between 1 and 100".into());
    }
    let mut profile = SystemProfile::load_or_probe();
    if let Some(v) = threads { profile.n_threads = v; }
    if let Some(v) = batch { profile.n_batch = v; }
    let physical = ubatch.unwrap_or(profile.n_batch.min(512));
    if profile.n_threads == 0 || profile.n_batch == 0 || physical == 0 || physical > profile.n_batch {
        return Err("threads and batch must be positive; ubatch must be within 1..=batch".into());
    }
    let mut dispatcher = RuntimeDispatcher::new();
    dispatcher.register(Box::new(LlamaCppBackend::new()));
    dispatcher.dispatch(RuntimeOp::Load {
        path: PathBuf::from(&model),
        n_ctx: profile.n_ctx,
        n_batch: profile.n_batch,
        n_ubatch: physical,
        n_threads: profile.n_threads,
    }).map_err(|e| format!("model load: {e}"))?;

    // The same separately tokenized segments are used in both candidates,
    // so the experiment measures KV reuse rather than BPE boundary changes.
    let first = tokenize(&mut dispatcher, &format!("User: {first_prompt}\nAssistant:"))?;
    let reply = tokenize(&mut dispatcher, &assistant_fragment)?;
    let second = tokenize(&mut dispatcher, &format!("\nUser: {second_prompt}\nAssistant:"))?;
    let full_len = first.len() + reply.len() + second.len();
    if full_len >= profile.n_ctx as usize {
        return Err(format!("{full_len} input tokens exceed the configured n_ctx={} (leave space for generation)", profile.n_ctx));
    }

    // Warm both modes, then alternate mode order on measured runs.
    evaluate(&mut dispatcher, &first, &reply, &second, false)?;
    evaluate(&mut dispatcher, &first, &reply, &second, true)?;
    let mut records = Vec::with_capacity(runs);
    for i in 0..runs {
        let (full_time, full_logits, cached_time, cached_logits) = if i % 2 == 0 {
            let (a, aa) = evaluate(&mut dispatcher, &first, &reply, &second, false)?;
            let (b, bb) = evaluate(&mut dispatcher, &first, &reply, &second, true)?;
            (a, aa, b, bb)
        } else {
            let (b, bb) = evaluate(&mut dispatcher, &first, &reply, &second, true)?;
            let (a, aa) = evaluate(&mut dispatcher, &first, &reply, &second, false)?;
            (a, aa, b, bb)
        };
        if full_logits.len() != cached_logits.len() {
            return Err("logit vector sizes differ between modes".into());
        }
        let full_top = top_token(&full_logits)
            .ok_or_else(|| "full-prefix returned empty/non-finite logits".to_string())?;
        let cached_top = top_token(&cached_logits)
            .ok_or_else(|| "cached path returned empty/non-finite logits".to_string())?;
        let max_abs_logit_delta = full_logits.iter().zip(&cached_logits)
            .map(|(a,b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max);
        records.push(serde_json::json!({
            "full_prefix_prefill_ms": full_time,
            "cached_second_turn_prefill_ms": cached_time,
            "argmax_equal": full_top == cached_top,
            "max_abs_logit_delta": max_abs_logit_delta,
        }));
    }
    let full_summary = summary(records.iter().map(|r| r["full_prefix_prefill_ms"].as_f64().unwrap()).collect());
    let cached_summary = summary(records.iter().map(|r| r["cached_second_turn_prefill_ms"].as_f64().unwrap()).collect());
    let report = serde_json::json!({
        "phase": "3.5",
        "measurement": "second_turn_prefill_reuse_vs_full_reprefill",
        "model": model,
        "context_tokens": full_len,
        "reused_prefix_tokens": first.len() + reply.len(),
        "new_second_turn_tokens": second.len(),
        "runs": runs,
        "threads": profile.n_threads,
        "n_batch": profile.n_batch,
        "n_ubatch": physical,
        "full_prefix_prefill_ms": full_summary,
        "cached_second_turn_prefill_ms": cached_summary,
        "all_argmax_equal": records.iter().all(|r| r["argmax_equal"] == true),
        "note": "Full-prefix prefill is compared with only the incremental second-turn prefill after an already evaluated first turn. This measures one warm-conversation step, not total inference cost or a 27 tok/s decode rate. Generic role delimiters are not a model-specific chat template.",
        "samples": records,
    });
    if json { println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?); }
    else {
        println!("KILN Phase 3.5 KV reuse A/B (same token sequence)");
        println!("  reused prefix tokens: {}", first.len()+reply.len());
        println!("  second-turn new tokens: {}", second.len());
        println!("  full prefix median: {:.2} ms", full_summary["median"].as_f64().unwrap());
        println!("  cached second-turn median: {:.2} ms", cached_summary["median"].as_f64().unwrap());
        println!("  same next-token argmax across runs: {}", report["all_argmax_equal"]);
        println!("  Note: not a decode tok/s benchmark or hardware acceptance.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_no_data_before_touching_model() {
        assert!(run(String::new(), String::new(), String::new(),
            String::new(), 0, None, None, None, false).is_err());
    }
    #[test]
    fn deterministic_summary() {
        let x = summary(vec![3.0, 1.0, 2.0]);
        assert_eq!(x["median"], 2.0);
        assert_eq!(x["min"], 1.0);
        assert_eq!(x["max"], 3.0);
    }
}
