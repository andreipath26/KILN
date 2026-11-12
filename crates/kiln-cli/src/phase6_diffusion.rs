//! Phase 6 masked-diffusion inference with Cluster-Level Attention-Guided
//! parallel decoding (CLAD). The native shim supplies LLaDA proposals,
//! probabilities, and attention averaged over the final four layers/heads.

use kiln_runtime::LlamaContext;
use std::path::Path;
use std::time::Instant;

const TIER0_CONTEXT_TOKENS: usize = 512;

pub fn run(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    requested_steps: Option<usize>,
    block_length: usize,
    threshold: f32,
    requested_threads: Option<usize>,
    json: bool,
) -> Result<(), String> {
    run_cases(
        model,
        &[(String::from("prompt-1"), prompt.to_owned())],
        max_tokens,
        requested_steps,
        block_length,
        threshold,
        requested_threads,
        json,
        "clad",
    )
}

pub fn run_cases(
    model: &str,
    cases: &[(String, String)],
    max_tokens: usize,
    requested_steps: Option<usize>,
    block_length: usize,
    threshold: f32,
    requested_threads: Option<usize>,
    json: bool,
    decoder: &str,
) -> Result<(), String> {
    if decoder != "clad" && decoder != "confidence" {
        return Err("--decoder must be 'clad' or 'confidence'".into());
    }
    if max_tokens == 0 || block_length == 0 || max_tokens % block_length != 0 {
        return Err("--max-tokens must be a positive multiple of --block-length".into());
    }
    let blocks = max_tokens / block_length;
    let steps = requested_steps.unwrap_or(max_tokens);
    if steps == 0 || steps % blocks != 0 {
        return Err("--steps must be positive and divisible by the number of generation blocks".into());
    }
    if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
        return Err("--threshold must be a finite probability in 0..=1".into());
    }
    if max_tokens >= TIER0_CONTEXT_TOKENS {
        return Err(format!(
            "--max-tokens must be below the Tier 0 context bound of {TIER0_CONTEXT_TOKENS}"
        ));
    }

    let threads = requested_threads
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .clamp(1, i32::MAX as usize) as i32;
    let model_path = Path::new(model);
    let mut ctx = LlamaContext::load_with_ubatch(
        model_path,
        Some((
            TIER0_CONTEXT_TOKENS as i32,
            TIER0_CONTEXT_TOKENS as i32,
            TIER0_CONTEXT_TOKENS as i32,
            threads,
        )),
    ).map_err(|e| format!("failed to load diffusion GGUF: {e}"))?;
    let shift_logits = ctx.diffusion_shift_logits()
        .map_err(|e| format!("failed to read LLaDA diffusion metadata: {e}"))?;

    if cases.is_empty() {
        return Err("prompt suite contains no cases".into());
    }

    for (case_id, prompt) in cases {
        let formatted = ctx.format_user_prompt(prompt)
            .map_err(|e| format!("{case_id}: failed to apply model chat template: {e}"))?;
        let input = ctx.tokenize(&formatted)
            .map_err(|e| format!("{case_id}: failed to tokenize formatted prompt: {e}"))?;
        if input.is_empty() {
            return Err(format!("{case_id}: chat template produced an empty token sequence"));
        }
        let total = input.len().checked_add(max_tokens)
            .ok_or_else(|| "prompt plus generation length overflow".to_string())?;
        if total > TIER0_CONTEXT_TOKENS {
            return Err(format!(
                "{case_id}: formatted prompt uses {} tokens; with {max_tokens} generated tokens it exceeds the {TIER0_CONTEXT_TOKENS}-token Phase 6 context limit",
                input.len()
            ));
        }
        let mask = ctx.mask_token()
            .map_err(|e| format!("{case_id}: Phase 6 requires a supported diffusion model: {e}"))?;
        let eos = ctx.eos_token();

        let mut sequence = Vec::with_capacity(total);
        sequence.extend_from_slice(&input);
        sequence.resize(total, mask);
        let steps_per_block = steps / blocks;
        let started = Instant::now();

        for block in 0..blocks {
            let start = input.len() + block * block_length;
            let end = start + block_length;
            for step in 0..steps_per_block {
                let masked: Vec<usize> = (start..end)
                    .filter(|&pos| sequence[pos] == mask)
                    .collect();
                if masked.is_empty() {
                    break;
                }

                let (proposals, confidence, attention) = ctx.diffusion_step(&sequence, decoder == "clad")
                    .map_err(|e| format!("{case_id}: native diffusion step failed: {e}"))?;
                let mut selected = if decoder == "clad" {
                    let mut candidates: Vec<usize> = masked.iter().copied()
                        .filter(|&pos| confidence[pos] >= threshold)
                        .collect();
                    if let Some(&best) = masked.iter().max_by(|&&a, &&b| {
                        confidence[a].total_cmp(&confidence[b]).then_with(|| b.cmp(&a))
                    }) {
                        if !candidates.contains(&best) {
                            candidates.push(best);
                        }
                    }
                    candidates.sort_unstable();

                    filter_clad_candidates(
                        &candidates, &attention, total, start, block_length,
                    )
                } else {
                    select_confidence_transfers(
                        &masked, &confidence, block_length, steps_per_block, step,
                    )
                };
                if selected.is_empty() {
                    if let Some(&best) = masked.iter().max_by(|&&a, &&b| {
                        confidence[a].total_cmp(&confidence[b]).then_with(|| b.cmp(&a))
                    }) {
                        selected.push(best);
                    }
                }
                for position in selected {
                    sequence[position] = proposals[position];
                }
            }
        }

        let remaining = sequence[input.len()..].iter().filter(|&&token| token == mask).count();
        if remaining != 0 {
            return Err(format!(
                "{case_id}: CLAD stopped with {remaining} masked positions; refusing to return an incomplete answer"
            ));
        }

        let elapsed = started.elapsed();
        let mut answer_bytes = Vec::new();
        let mut generated_tokens = 0usize;
        for &token in &sequence[input.len()..] {
            if token == eos {
                break;
            }
            answer_bytes.extend_from_slice(
                &ctx.token_bytes(token).map_err(|e| format!("{case_id}: token decode failed: {e}"))?
            );
            generated_tokens += 1;
        }
        let answer = String::from_utf8_lossy(&answer_bytes).into_owned();
        let elapsed_seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        let tok_per_second = generated_tokens as f64 / elapsed_seconds;

        if json {
            println!("{}", serde_json::json!({
                "phase": 6,
                "case_id": case_id,
                "method": if decoder == "clad" { "CLAD" } else { "llama.cpp-confidence" },
                "decoder": decoder,
                "model": model,
                "threads": threads,
                "shift_logits": shift_logits,
                "input_tokens": input.len(),
                "generated_tokens": generated_tokens,
                "steps": steps,
                "steps_per_block": steps_per_block,
                "block_length": block_length,
                "confidence_threshold": threshold,
                "elapsed_ms": elapsed.as_millis(),
                "decode_tokens_per_second": tok_per_second,
                "answer": answer,
                "peak_rss_kib": super::process_memory_kib().map(|m| m.peak_rss_kib),
                "hardware_acceptance": false
            }));
        } else {
            println!("{answer}");
            eprintln!(
                "KILN Phase 6 {decoder}: {generated_tokens} tokens in {:.2}s ({tok_per_second:.2} tok/s); {steps} steps, block {block_length}, threshold {threshold:.2}",
                elapsed_seconds
            );
        }

    }
    Ok(())
}

/// Select the standard greedy confidence-based transfer quota for one diffusion step.
fn select_confidence_transfers(
    masked: &[usize],
    confidence: &[f32],
    block_length: usize,
    steps_per_block: usize,
    step: usize,
) -> Vec<usize> {
    if steps_per_block == 0 || step >= steps_per_block
        || masked.iter().any(|&position| position >= confidence.len())
    {
        return Vec::new();
    }
    let transfer_count = block_length / steps_per_block
        + if step < block_length % steps_per_block { 1 } else { 0 };
    let mut ordered = masked.to_vec();
    ordered.sort_by(|&a, &b| {
        confidence[b].total_cmp(&confidence[a]).then_with(|| a.cmp(&b))
    });
    ordered.truncate(transfer_count.min(masked.len()));
    ordered
}

/// Apply CLAD's cluster dependency filter to the currently proposed positions.
/// attention is row-major [query][key] for the full input+generation sequence.
fn filter_clad_candidates(
    candidates: &[usize],
    attention: &[f32],
    sequence_len: usize,
    block_start: usize,
    block_len: usize,
) -> Vec<usize> {
    if candidates.is_empty()
        || sequence_len == 0
        || attention.len() != sequence_len.saturating_mul(sequence_len)
        || block_start.checked_add(block_len).map_or(true, |end| end > sequence_len)
    {
        return Vec::new();
    }
    let mut local: Vec<usize> = candidates.iter()
        .copied()
        .filter(|&p| p >= block_start && p < block_start + block_len)
        .map(|p| p - block_start)
        .collect();
    local.sort_unstable();
    local.dedup();
    if local.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut first = local[0];
    let mut last = first;
    for &position in &local[1..] {
        if position == last + 1 {
            last = position;
        } else {
            clusters.push((first, last));
            first = position;
            last = position;
        }
    }
    clusters.push((first, last));
    let count = clusters.len();
    if count == 1 {
        return (clusters[0].0..=clusters[0].1)
            .map(|p| block_start + p)
            .collect();
    }

    let mut score = vec![0.0f32; count * count];
    for i in 0..count {
        for j in (i + 1)..count {
            let (i_start, i_end) = clusters[i];
            let (j_start, j_end) = clusters[j];
            let mut best = 0.0f32;
            for query in i_start..=i_end {
                let mut mean = 0.0f32;
                for key in j_start..=j_end {
                    let a = attention[(block_start + query) * sequence_len + block_start + key];
                    let b = attention[(block_start + key) * sequence_len + block_start + query];
                    let ab = if a.is_finite() { a.max(0.0) } else { 0.0 };
                    let ba = if b.is_finite() { b.max(0.0) } else { 0.0 };
                    mean += ab.max(ba);
                }
                mean /= (j_end - j_start + 1) as f32;
                best = best.max(mean);
            }
            score[i * count + j] = best;
            score[j * count + i] = best;
        }
    }

    let mut row_mean = vec![0.0f32; count];
    let mut top1 = vec![0usize; count];
    for i in 0..count {
        let mut best_j = if i == 0 { 1 } else { 0 };
        let mut sum = 0.0f32;
        for j in 0..count {
            if i == j { continue; }
            let value = score[i * count + j];
            sum += value;
            if value > score[i * count + best_j] || best_j == i {
                best_j = j;
            }
        }
        top1[i] = best_j;
        row_mean[i] = sum / (count - 1) as f32;
    }

    let mut selected_cluster = vec![false; count];
    let mut conflict_graph = vec![false; count * count];
    for i in 0..count {
        let j = top1[i];
        if top1[j] != i {
            continue;
        }
        let edge = score[i * count + j];
        let is_conflict = if count <= 2 {
            edge >= 1.2 / block_len.max(1) as f32
        } else {
            edge >= row_mean[i] && edge >= row_mean[j]
        };
        if !is_conflict {
            continue;
        }
        conflict_graph[i * count + j] = true;
        conflict_graph[j * count + i] = true;
        let weight_i = clusters[i].1 - clusters[i].0 + 1;
        let weight_j = clusters[j].1 - clusters[j].0 + 1;
        if weight_i >= weight_j {
            selected_cluster[i] = true;
        } else {
            selected_cluster[j] = true;
        }
    }
    for i in 0..count {
        if (0..count).all(|j| !conflict_graph[i * count + j]) {
            selected_cluster[i] = true;
        }
    }
    if selected_cluster.iter().all(|selected| !selected) {
        let best = (0..count).max_by_key(|&i| clusters[i].1 - clusters[i].0 + 1).unwrap_or(0);
        selected_cluster[best] = true;
    }

    let mut result = Vec::new();
    for (i, &(start, end)) in clusters.iter().enumerate() {
        if selected_cluster[i] {
            result.extend((start..=end).map(|p| block_start + p));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::filter_clad_candidates;

    #[test]
    fn confidence_decoder_uses_even_blockwise_transfer_quotas_and_top_confidence() {
        let masked: Vec<usize> = (0..8).collect();
        let confidence = vec![0.1, 0.2, 0.9, 0.4, 0.7, 0.3, 0.8, 0.5];
        assert_eq!(
            super::select_confidence_transfers(&masked, &confidence, 8, 4, 0),
            vec![2, 6]
        );
        let remaining = vec![0, 1, 3, 4, 5, 7];
        assert_eq!(
            super::select_confidence_transfers(&remaining, &confidence, 8, 4, 1),
            vec![4, 7]
        );
    }

    #[test]
    fn confidence_decoder_distributes_odd_quota_to_early_steps() {
        let masked: Vec<usize> = (0..5).collect();
        let confidence = vec![0.2, 0.9, 0.1, 0.8, 0.3];
        assert_eq!(
            super::select_confidence_transfers(&masked, &confidence, 5, 3, 0).len(),
            2
        );
        assert_eq!(
            super::select_confidence_transfers(&masked, &confidence, 5, 3, 1).len(),
            2
        );
        assert_eq!(
            super::select_confidence_transfers(&masked, &confidence, 5, 3, 2).len(),
            1
        );
    }

    #[test]
    fn independent_candidates_are_committed_together() {
        let attention = vec![0.0; 8 * 8];
        let selected = filter_clad_candidates(&[2, 5], &attention, 8, 0, 8);
        assert_eq!(selected, vec![2, 5]);
    }

    #[test]
    fn attention_conflict_prefers_the_larger_contiguous_cluster() {
        let mut attention = vec![0.0; 8 * 8];
        attention[0 * 8 + 4] = 0.8;
        attention[4 * 8 + 0] = 0.8;
        let selected = filter_clad_candidates(&[0, 1, 4], &attention, 8, 0, 8);
        assert_eq!(selected, vec![0, 1]);
    }

    #[test]
    fn one_cluster_is_preserved_without_attention_conflicts() {
        let attention = vec![0.0; 8 * 8];
        let selected = filter_clad_candidates(&[3, 4, 5], &attention, 8, 0, 8);
        assert_eq!(selected, vec![3, 4, 5]);
    }
}
