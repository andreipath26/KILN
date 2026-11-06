//! Phase 5.0: leakage-resistant, chronological expert predictor feasibility.
//!
//! This evaluates the existing bounded one-step expert transition planner.
//! It does not train a neural prerouter or observe real model routing by itself.
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::expert_cache::ExpertKey;
use crate::expert_prefetch::ExpertPrefetcher;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictorHoldoutReport {
    pub train_events: usize,
    pub test_events: usize,
    pub top_k: usize,
    pub trained_distinct_experts: usize,
    pub predicted_events: usize,
    pub correct_events: usize,
    /// Correct / all held-out events; unpredicted events count as misses.
    pub accuracy_at_k: f64,
    /// Events with at least one candidate / held-out events.
    pub coverage: f64,
    /// Correct / predicted events. This is not accuracy over all events.
    pub precision_when_predicted: f64,
    /// Global train-only most-frequent-expert top-k baseline, same test events.
    pub unigram_baseline_accuracy_at_k: f64,
    pub passed_80_percent_feasibility_threshold: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldoutError {
    InvalidSplit,
    InvalidTopK,
}
impl std::fmt::Display for HoldoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HoldoutError {}

/// Train ONLY on the chronological prefix, then evaluate without observing
/// or adapting to the held-out suffix. The test trace provides no candidate
/// sizes, hotness classes or transition statistics. Candidate keys come from
/// training history only. This is a single-expert next-event diagnostic:
/// real multi-expert-per-token routing needs a separate evaluation protocol.
pub fn evaluate_predictor_holdout(
    trace: &[ExpertKey], train_events: usize, top_k: usize,
) -> Result<PredictorHoldoutReport, HoldoutError> {
    if train_events < 2 || train_events >= trace.len() {
        return Err(HoldoutError::InvalidSplit);
    }
    if !(1..=16).contains(&top_k) {
        return Err(HoldoutError::InvalidTopK);
    }
    let mut predictor = ExpertPrefetcher::new(top_k, u64::MAX);
    let mut global_counts: HashMap<ExpertKey, usize> = HashMap::new();
    let mut unique: HashSet<ExpertKey> = HashSet::new();
    for &key in &trace[..train_events] {
        predictor.observe(key);
        unique.insert(key);
        *global_counts.entry(key).or_insert(0) += 1;
    }
    let mut candidates: Vec<(ExpertKey, u64)> =
        unique.iter().map(|&key| (key, 1)).collect();
    candidates.sort_unstable_by_key(|(key, _)| (key.layer, key.expert_id));
    let mut baseline: Vec<(ExpertKey, usize)> = global_counts.into_iter().collect();
    baseline.sort_by(|a, b| b.1.cmp(&a.1)
        .then_with(|| a.0.layer.cmp(&b.0.layer))
        .then_with(|| a.0.expert_id.cmp(&b.0.expert_id)));
    let global_top: HashSet<ExpertKey> = baseline.into_iter()
        .take(top_k).map(|(key, _)| key).collect();

    let mut predicted = 0usize;
    let mut correct = 0usize;
    let mut baseline_correct = 0usize;
    for i in train_events..trace.len() {
        let source = trace[i-1];
        let actual = trace[i];
        let proposals = predictor.predict(source, &candidates, top_k);
        if !proposals.is_empty() {
            predicted += 1;
            if proposals.iter().any(|p| p.key == actual) {
                correct += 1;
            }
        }
        if global_top.contains(&actual) {
            baseline_correct += 1;
        }
        // clear_pending releases planner bookkeeping; it does NOT observe
        // actual test labels or mutate transition frequencies.
        predictor.clear_pending();
    }
    let test = trace.len() - train_events;
    let accuracy = correct as f64 / test as f64;
    Ok(PredictorHoldoutReport {
        train_events, test_events: test, top_k,
        trained_distinct_experts: unique.len(),
        predicted_events: predicted,
        correct_events: correct,
        accuracy_at_k: accuracy,
        coverage: predicted as f64 / test as f64,
        precision_when_predicted: if predicted == 0 { 0.0 } else {
            correct as f64 / predicted as f64
        },
        unigram_baseline_accuracy_at_k: baseline_correct as f64 / test as f64,
        passed_80_percent_feasibility_threshold: accuracy >= 0.8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(i: u32) -> ExpertKey { ExpertKey { layer: 0, expert_id: i } }

    #[test]
    fn repeating_transition_passes_on_disjoint_holdout() {
        let trace: Vec<_> = (0..20).flat_map(|_| [key(0), key(1)]).collect();
        let report = evaluate_predictor_holdout(&trace, 20, 2).unwrap();
        assert_eq!(report.test_events, 20);
        assert_eq!(report.correct_events, 20);
        assert!(report.passed_80_percent_feasibility_threshold);
        assert_eq!(report.coverage, 1.0);
    }

    #[test]
    fn unseen_holdout_never_leaks_into_training() {
        let mut trace: Vec<_> = (0..20).flat_map(|_| [key(0), key(1)]).collect();
        trace.extend(std::iter::repeat_n(key(99), 10));
        let report = evaluate_predictor_holdout(&trace, 40, 2).unwrap();
        assert_eq!(report.correct_events, 0);
        assert_eq!(report.accuracy_at_k, 0.0);
        assert_eq!(report.unigram_baseline_accuracy_at_k, 0.0);
        assert!(!report.passed_80_percent_feasibility_threshold);
    }

    #[test]
    fn insufficient_history_and_invalid_k_fail() {
        let data = [key(0), key(1), key(2)];
        assert_eq!(evaluate_predictor_holdout(&data, 1, 2).unwrap_err(), HoldoutError::InvalidSplit);
        assert_eq!(evaluate_predictor_holdout(&data, 3, 2).unwrap_err(), HoldoutError::InvalidSplit);
        assert_eq!(evaluate_predictor_holdout(&data, 2, 0).unwrap_err(), HoldoutError::InvalidTopK);
    }

    #[test]
    fn tie_breaks_are_deterministic() {
        let trace = [key(0), key(2), key(0), key(1), key(0), key(2), key(0), key(1)];
        let first = evaluate_predictor_holdout(&trace, 6, 1).unwrap();
        let second = evaluate_predictor_holdout(&trace, 6, 1).unwrap();
        assert_eq!(first, second);
    }
}
