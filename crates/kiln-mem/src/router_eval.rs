//! Phase 5.0 grouped MoE router-event feasibility evaluation.
//!
//! Unlike a flat expert-access stream, each event records the entire expert
//! set selected by one layer at one token. Training never sees held-out labels.
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RouterEvent {
    pub token_position: u64,
    pub layer: u32,
    pub active_experts: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroupedHoldoutReport {
    pub train_events: usize,
    pub test_events: usize,
    pub top_k: usize,
    pub covered_events: usize,
    pub correct_expert_sets: usize,
    /// All active target experts included in the top-k prediction.
    pub accuracy_at_k: f64,
    pub coverage: f64,
    /// Fraction of individually activated experts recovered.
    pub active_expert_recall_at_k: f64,
    /// Training-only global expert-frequency baseline, evaluated per layer.
    pub unigram_baseline_accuracy_at_k: f64,
    pub exploratory_threshold_80_percent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterEvalError {
    InvalidSplit,
    InvalidTopK,
    UnsortedOrDuplicateEvent,
    InvalidExpertSelection,
}

impl std::fmt::Display for RouterEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RouterEvalError {}

type TransitionCounts = HashMap<(u32, u32), HashMap<u32, u64>>;
type ExpertFreq = HashMap<u32, HashMap<u32, u64>>;

fn rank_candidates(
    prev: &[u32], layer: u32, transitions: &TransitionCounts, k: usize,
) -> Vec<u32> {
    let mut totals: HashMap<u32, u64> = HashMap::new();
    for &previous in prev {
        if let Some(targets) = transitions.get(&(layer, previous)) {
            for (&target, &count) in targets {
                let total = totals.entry(target).or_default();
                *total = total.saturating_add(count);
            }
        }
    }
    let mut ranked: Vec<_> = totals.into_iter().collect();
    ranked.sort_unstable_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(k).map(|(expert,_)| expert).collect()
}

fn baseline_candidates(layer: u32, frequency: &ExpertFreq, k: usize) -> Vec<u32> {
    let mut ranked: Vec<_> = frequency.get(&layer)
        .map(|counts| counts.iter().map(|(&key,&value)| (key,value)).collect())
        .unwrap_or_else(Vec::new);
    ranked.sort_unstable_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(k).map(|(expert,_)| expert).collect()
}

/// Split by token position, never within a token across layers. Previous
/// held-out router decisions become causal context for later predictions,
/// but do NOT update the frozen trained transition statistics.
pub fn evaluate_grouped_router_holdout(
    events: &[RouterEvent], train_until_token: u64, top_k: usize,
) -> Result<GroupedHoldoutReport, RouterEvalError> {
    if !(1..=16).contains(&top_k) {
        return Err(RouterEvalError::InvalidTopK);
    }
    if events.is_empty() {
        return Err(RouterEvalError::InvalidSplit);
    }
    for (i,event) in events.iter().enumerate() {
        let valid = !event.active_experts.is_empty()
            && event.active_experts.len() <= top_k
            && event.active_experts.iter().copied().collect::<HashSet<_>>().len()
                == event.active_experts.len();
        if !valid {
            return Err(RouterEvalError::InvalidExpertSelection);
        }
        if i > 0 {
            let prior=&events[i-1];
            if (event.token_position,event.layer) <= (prior.token_position,prior.layer) {
                return Err(RouterEvalError::UnsortedOrDuplicateEvent);
            }
        }
    }
    let split=events.partition_point(|event| event.token_position < train_until_token);
    if split < 2 || split == events.len() {
        return Err(RouterEvalError::InvalidSplit);
    }

    let mut transitions: TransitionCounts = HashMap::new();
    let mut frequencies: ExpertFreq = HashMap::new();
    let mut previous: HashMap<u32, Vec<u32>> = HashMap::new();
    for event in &events[..split] {
        let layer_counts=frequencies.entry(event.layer).or_default();
        for &target in &event.active_experts {
            let count=layer_counts.entry(target).or_default();
            *count=count.saturating_add(1);
        }
        if let Some(prev)=previous.get(&event.layer) {
            for &source in prev {
                let next=transitions.entry((event.layer,source)).or_default();
                for &target in &event.active_experts {
                    let count=next.entry(target).or_default();
                    *count=count.saturating_add(1);
                }
            }
        }
        previous.insert(event.layer, event.active_experts.clone());
    }
    let mut covered=0usize;
    let mut correct=0usize;
    let mut baseline_correct=0usize;
    let mut recovered=0usize;
    let mut active_total=0usize;
    for event in &events[split..] {
        let proposed=previous.get(&event.layer)
            .map(|prev| rank_candidates(prev,event.layer,&transitions,top_k))
            .unwrap_or_default();
        if !proposed.is_empty() { covered+=1; }
        let all_hit=event.active_experts.iter().all(|id| proposed.contains(id));
        if all_hit && !proposed.is_empty() { correct+=1; }
        let b=baseline_candidates(event.layer,&frequencies,top_k);
        if !b.is_empty() && event.active_experts.iter().all(|id| b.contains(id)) {
            baseline_correct+=1;
        }
        recovered+=event.active_experts.iter().filter(|id| proposed.contains(id)).count();
        active_total+=event.active_experts.len();
        // Knowing the *past* routed experts is legitimate causal context.
        // Never add heldout transitions to the model under evaluation.
        previous.insert(event.layer,event.active_experts.clone());
    }
    let test=events.len()-split;
    let acc=correct as f64 / test as f64;
    Ok(GroupedHoldoutReport {
        train_events:split,test_events:test,top_k,
        covered_events:covered,correct_expert_sets:correct,
        accuracy_at_k:acc,coverage:covered as f64/test as f64,
        active_expert_recall_at_k:recovered as f64/active_total as f64,
        unigram_baseline_accuracy_at_k:baseline_correct as f64/test as f64,
        exploratory_threshold_80_percent:acc>=0.8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn event(t:u64,l:u32,a:&[u32])->RouterEvent {
        RouterEvent{token_position:t,layer:l,active_experts:a.to_vec()}
    }
    #[test]
    fn reconstructs_multi_expert_sets_from_training_only() {
        let events:Vec<_>=(0..20).map(|i|event(i,0,
            if i%2==0 {&[0,1]} else {&[2,3]})).collect();
        let r=evaluate_grouped_router_holdout(&events,10,2).unwrap();
        assert_eq!(r.correct_expert_sets,10);
        assert_eq!(r.coverage,1.0);
        assert_eq!(r.active_expert_recall_at_k,1.0);
        assert!(r.exploratory_threshold_80_percent);
        assert!(r.unigram_baseline_accuracy_at_k < r.accuracy_at_k);
    }
    #[test]
    fn test_labels_do_not_update_frozen_transition_counts() {
        let mut events:Vec<_>=(0..10).map(|i|event(i,0,
            if i%2==0 {&[0,1]} else {&[2,3]})).collect();
        events.extend((10..20).map(|i|event(i,0,&[90,91])));
        let r=evaluate_grouped_router_holdout(&events,10,2).unwrap();
        assert_eq!(r.correct_expert_sets,0);
        assert!(!r.exploratory_threshold_80_percent);
    }
    #[test]
    fn rejects_unsafe_event_shapes() {
        assert_eq!(evaluate_grouped_router_holdout(
            &[event(0,0,&[1,1]),event(1,0,&[1,2])],1,2).unwrap_err(),
            RouterEvalError::InvalidExpertSelection);
        assert_eq!(evaluate_grouped_router_holdout(
            &[event(0,0,&[1]),event(0,0,&[2]),event(1,0,&[3])],1,2).unwrap_err(),
            RouterEvalError::UnsortedOrDuplicateEvent);
        assert_eq!(evaluate_grouped_router_holdout(
            &[event(0,0,&[1,2,3]),event(1,0,&[1,2])],1,2).unwrap_err(),
            RouterEvalError::InvalidExpertSelection);
    }
    #[test]
    fn separates_layers_even_at_identical_token_position() {
        let mut events=Vec::new();
        for t in 0..20 {
            events.push(event(t,0,if t%2==0 {&[0,1]} else {&[2,3]}));
            events.push(event(t,1,if t%2==0 {&[10,11]} else {&[12,13]}));
        }
        let r=evaluate_grouped_router_holdout(&events,10,2).unwrap();
        assert_eq!(r.test_events,20);
        assert_eq!(r.correct_expert_sets,20);
    }
}
