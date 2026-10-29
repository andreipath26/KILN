//! Trace-driven, bounded expert prefetch prediction primitives.
//!
//! This module plans candidates only. It does not start I/O, does not observe
//! live llama.cpp routing, and does not claim that a planned read overlapped
//! compute. The caller must execute, cancel, and measure these plans.
use std::collections::{HashMap, VecDeque};

use crate::expert_cache::ExpertKey;

/// A candidate returned by the predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchCandidate {
    pub key: ExpertKey,
    /// Empirical conditional transition frequency in [0, 1].
    pub confidence: f64,
    pub expected_bytes: u64,
}

/// Bounded counters for prefetch plans and caller-reported outcomes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    pub predictions: u64,
    pub scheduled: u64,
    pub rejected_request_cap: u64,
    pub rejected_byte_cap: u64,
    pub useful: u64,
    pub late: u64,
    pub unused: u64,
    pub cancelled: u64,
    pub in_flight_requests: usize,
    pub in_flight_bytes: u64,
}

/// Learns one-step transitions from an observed expert-key sequence and emits
/// confidence-ranked plans under request and byte limits.
pub struct ExpertPrefetcher {
    transitions: HashMap<ExpertKey, HashMap<ExpertKey, u64>>,
    totals: HashMap<ExpertKey, u64>,
    previous: Option<ExpertKey>,
    max_pending_requests: usize,
    max_in_flight_bytes: u64,
    in_flight: HashMap<ExpertKey, u64>,
    ready: HashMap<ExpertKey, u64>,
    order: VecDeque<ExpertKey>,
    stats: PrefetchStats,
}

impl ExpertPrefetcher {
    pub fn new(max_pending_requests: usize, max_in_flight_bytes: u64) -> Self {
        Self {
            transitions: HashMap::new(),
            totals: HashMap::new(),
            previous: None,
            max_pending_requests,
            max_in_flight_bytes,
            in_flight: HashMap::new(),
            ready: HashMap::new(),
            order: VecDeque::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Observe an actual expert access. Training is online and deterministic.
    pub fn observe(&mut self, actual: ExpertKey) {
        if let Some(previous) = self.previous {
            *self.transitions.entry(previous).or_default().entry(actual).or_default() += 1;
            *self.totals.entry(previous).or_default() += 1;
        }
        self.previous = Some(actual);
        if self.ready.remove(&actual).is_some() {
            self.stats.useful = self.stats.useful.saturating_add(1);
        } else if self.in_flight.remove(&actual).is_some() {
            // The expert was demanded before the caller reported the prefetch
            // as ready, so it did not hide the miss.
            self.order.retain(|key| *key != actual);
            self.refresh_in_flight();
            self.stats.late = self.stats.late.saturating_add(1);
        }
    }

    /// Rank possible next expert keys using the observed transition history.
    /// Candidate sizes are supplied by the model index; unknown keys are ignored.
    pub fn predict(
        &mut self,
        current: ExpertKey,
        candidate_sizes: &[(ExpertKey, u64)],
        top_k: usize,
    ) -> Vec<PrefetchCandidate> {
        self.stats.predictions = self.stats.predictions.saturating_add(1);
        if top_k == 0 || self.max_pending_requests == 0 || self.max_in_flight_bytes == 0 {
            return Vec::new();
        }
        let Some(total) = self.totals.get(&current).copied().filter(|v| *v > 0) else {
            return Vec::new();
        };
        let Some(counts) = self.transitions.get(&current) else {
            return Vec::new();
        };
        let mut ranked: Vec<_> = candidate_sizes.iter()
            .filter_map(|(key, bytes)| {
                let count = counts.get(key).copied()?;
                if *bytes == 0 || self.in_flight.contains_key(key) {
                    return None;
                }
                Some(PrefetchCandidate {
                    key: *key,
                    confidence: count as f64 / total as f64,
                    expected_bytes: *bytes,
                })
            })
            .collect();
        ranked.sort_by(|a, b| b.confidence.total_cmp(&a.confidence)
            .then_with(|| a.expected_bytes.cmp(&b.expected_bytes))
            .then_with(|| a.key.layer.cmp(&b.key.layer))
            .then_with(|| a.key.expert_id.cmp(&b.key.expert_id)));

        let mut planned = Vec::new();
        for candidate in ranked.into_iter().take(top_k) {
            if self.in_flight.len() + self.ready.len() >= self.max_pending_requests {
                self.stats.rejected_request_cap = self.stats.rejected_request_cap.saturating_add(1);
                break;
            }
            let used = self.stats.in_flight_bytes;
            if candidate.expected_bytes > self.max_in_flight_bytes.saturating_sub(used) {
                self.stats.rejected_byte_cap = self.stats.rejected_byte_cap.saturating_add(1);
                continue;
            }
            self.in_flight.insert(candidate.key, candidate.expected_bytes);
            self.order.push_back(candidate.key);
            self.stats.scheduled = self.stats.scheduled.saturating_add(1);
            self.refresh_in_flight();
            planned.push(candidate);
        }
        planned
    }

    /// Report that a prefetched read completed before the expert was demanded.
    /// Ready bytes are now expected to be owned/accounted for by the cache layer.
    pub fn mark_ready(&mut self, key: ExpertKey) -> bool {
        let Some(bytes) = self.in_flight.remove(&key) else {
            return false;
        };
        self.order.retain(|item| *item != key);
        self.ready.insert(key, bytes);
        self.refresh_in_flight();
        true
    }

    /// Report that a planned request completed after its consumer needed it.
    pub fn mark_late(&mut self, key: ExpertKey) -> bool {
        if self.in_flight.remove(&key).is_none() {
            return false;
        }
        self.order.retain(|item| *item != key);
        self.refresh_in_flight();
        self.stats.late = self.stats.late.saturating_add(1);
        true
    }

    /// Report that a prefetched block was never used.
    pub fn mark_unused(&mut self, key: ExpertKey) -> bool {
        let removed_in_flight = self.in_flight.remove(&key).is_some();
        let removed_ready = self.ready.remove(&key).is_some();
        if !removed_in_flight && !removed_ready {
            return false;
        }
        self.order.retain(|item| *item != key);
        self.refresh_in_flight();
        self.stats.unused = self.stats.unused.saturating_add(1);
        true
    }

    /// Cancel a plan that became stale. Returns false for unknown keys.
    pub fn cancel(&mut self, key: ExpertKey) -> bool {
        let removed_in_flight = self.in_flight.remove(&key).is_some();
        let removed_ready = self.ready.remove(&key).is_some();
        if !removed_in_flight && !removed_ready {
            return false;
        }
        self.order.retain(|item| *item != key);
        self.refresh_in_flight();
        self.stats.cancelled = self.stats.cancelled.saturating_add(1);
        true
    }

    /// Explicitly clear all pending plans and tracked ready candidates, e.g.
    /// when the prompt/session changes. Ready expert bytes themselves belong
    /// to the cache layer and are not evicted by this planner.
    pub fn clear_pending(&mut self) -> usize {
        let removed = self.in_flight.len() + self.ready.len();
        self.in_flight.clear();
        self.ready.clear();
        self.order.clear();
        self.refresh_in_flight();
        self.stats.cancelled = self.stats.cancelled.saturating_add(removed as u64);
        removed
    }

    pub fn stats(&self) -> PrefetchStats {
        self.stats
    }

    fn refresh_in_flight(&mut self) {
        self.stats.in_flight_requests = self.in_flight.len();
        self.stats.in_flight_bytes = self.in_flight.values().copied().sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: u32) -> ExpertKey {
        ExpertKey { layer: 0, expert_id: id }
    }

    #[test]
    fn predicts_most_common_transition_with_empirical_confidence() {
        let mut p = ExpertPrefetcher::new(4, 1024);
        for next in [1, 1, 1, 2] {
            p.observe(key(0));
            p.observe(key(next));
        }
        let plans = p.predict(key(0), &[(key(1), 100), (key(2), 200)], 2);
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].key, key(1));
        assert!((plans[0].confidence - 0.75).abs() < f64::EPSILON);
        assert_eq!(p.stats().in_flight_requests, 2);
        assert_eq!(p.stats().in_flight_bytes, 300);
    }

    #[test]
    fn enforces_request_and_byte_caps_and_ignores_zero_sized_candidates() {
        let mut p = ExpertPrefetcher::new(1, 128);
        for next in [1, 2, 1, 2] {
            p.observe(key(0));
            p.observe(key(next));
        }
        let plans = p.predict(key(0), &[(key(1), 64), (key(2), 256), (key(3), 0)], 3);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].key, key(1));
        assert_eq!(p.stats().in_flight_requests, 1);
        assert_eq!(p.stats().in_flight_bytes, 64);
        assert!(p.stats().rejected_request_cap > 0);
    }

    #[test]
    fn byte_cap_rejects_candidate_without_exceeding_budget() {
        let mut p = ExpertPrefetcher::new(4, 128);
        for next in [1, 2, 1, 2] {
            p.observe(key(0));
            p.observe(key(next));
        }
        let plans = p.predict(key(0), &[(key(1), 64), (key(2), 128)], 2);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].key, key(1));
        assert_eq!(p.stats().in_flight_bytes, 64);
        assert_eq!(p.stats().rejected_byte_cap, 1);
        assert!(p.stats().in_flight_bytes <= 128);
    }

    #[test]
    fn outcomes_and_cancellation_release_capacity() {
        let mut p = ExpertPrefetcher::new(3, 300);
        for next in [1, 2, 1, 2, 3] {
            p.observe(key(0));
            p.observe(key(next));
        }
        let candidates = [(key(1), 100), (key(2), 100), (key(3), 100)];
        assert_eq!(p.predict(key(0), &candidates, 3).len(), 3);
        assert!(p.mark_ready(key(1)));
        p.observe(key(1));
        assert!(p.mark_late(key(2)));
        assert!(p.mark_unused(key(3)));
        assert_eq!(p.stats().useful, 1);
        assert_eq!(p.stats().late, 1);
        assert_eq!(p.stats().unused, 1);
        assert_eq!(p.stats().in_flight_requests, 0);
        assert!(!p.cancel(key(99)));
    }

    #[test]
    fn clear_pending_cancels_all_and_prediction_without_history_is_empty() {
        let mut p = ExpertPrefetcher::new(2, 200);
        assert!(p.predict(key(0), &[(key(1), 100)], 1).is_empty());
        for next in [1, 2, 1] {
            p.observe(key(0));
            p.observe(key(next));
        }
        assert_eq!(p.predict(key(0), &[(key(1), 100), (key(2), 100)], 2).len(), 2);
        assert_eq!(p.clear_pending(), 2);
        assert_eq!(p.stats().in_flight_bytes, 0);
        assert_eq!(p.stats().cancelled, 2);
    }
}
