//! Phase 5.2: byte-bounded, token-scoped speculative expert Commit Guard.
//! Only actual model-router choices may authorize a prefetched expert.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::ExpertKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageResult { Staged, AlreadyStaged, InvalidEpoch, Empty, EntryLimit, ByteLimit }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitStats {
    pub staged: u64,
    pub accepted: u64,
    pub missed: u64,
    pub discarded: u64,
    pub bytes: u64,
}

pub struct CommitResult {
    pub accepted: Vec<(ExpertKey, Arc<[u8]>)>,
    pub missing: Vec<ExpertKey>,
}

pub struct ExpertCommitGuard {
    epoch: u64,
    max_bytes: u64,
    max_entries: usize,
    blocks: HashMap<ExpertKey, Arc<[u8]>>,
    stats: CommitStats,
}

impl ExpertCommitGuard {
    pub fn new(max_bytes: u64, max_entries: usize) -> Self {
        Self { epoch: 0, max_bytes, max_entries, blocks: HashMap::new(), stats: CommitStats::default() }
    }

    pub fn begin_token(&mut self, epoch: u64) -> Result<(), &'static str> {
        if epoch <= self.epoch { return Err("token epoch must increase"); }
        self.cancel();
        self.epoch = epoch;
        Ok(())
    }

    pub fn stage(&mut self, epoch: u64, key: ExpertKey, bytes: Arc<[u8]>) -> StageResult {
        if epoch == 0 || epoch != self.epoch { return StageResult::InvalidEpoch; }
        if bytes.is_empty() { return StageResult::Empty; }
        if self.blocks.contains_key(&key) { return StageResult::AlreadyStaged; }
        if self.blocks.len() >= self.max_entries { return StageResult::EntryLimit; }
        if bytes.len() as u64 > self.max_bytes.saturating_sub(self.stats.bytes) {
            return StageResult::ByteLimit;
        }
        self.stats.bytes += bytes.len() as u64;
        self.blocks.insert(key, bytes);
        self.stats.staged += 1;
        StageResult::Staged
    }

    /// Return only exact actual-router experts, preserving its order.
    /// Missing keys must be retrieved from authoritative model storage.
    pub fn commit(&mut self, epoch: u64, actual: &[ExpertKey]) -> Result<CommitResult, &'static str> {
        if epoch != self.epoch || epoch == 0 || actual.is_empty() {
            return Err("missing route or wrong epoch");
        }
        let mut seen = HashSet::new();
        if !actual.iter().all(|key| seen.insert(*key)) {
            return Err("duplicate actual expert");
        }
        let mut accepted = Vec::new();
        let mut missing = Vec::new();
        for &key in actual {
            if let Some(bytes) = self.blocks.remove(&key) {
                self.stats.bytes -= bytes.len() as u64;
                self.stats.accepted += 1;
                accepted.push((key, bytes));
            } else {
                self.stats.missed += 1;
                missing.push(key);
            }
        }
        self.cancel();
        Ok(CommitResult { accepted, missing })
    }

    pub fn cancel(&mut self) {
        self.stats.discarded += self.blocks.len() as u64;
        self.blocks.clear();
        self.stats.bytes = 0;
    }

    pub fn stats(&self) -> CommitStats { self.stats }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(id: u32) -> ExpertKey { ExpertKey { layer: 4, expert_id: id } }
    fn block(n: usize) -> Arc<[u8]> { Arc::from(vec![3u8; n]) }

    #[test]
    fn exact_router_selection_preserves_order_and_discards_wrong_predictions() {
        let mut guard = ExpertCommitGuard::new(100, 3);
        guard.begin_token(1).unwrap();
        for id in [1, 2, 9] {
            assert_eq!(guard.stage(1, key(id), block(20)), StageResult::Staged);
        }
        let result = guard.commit(1, &[key(2), key(3), key(1)]).unwrap();
        assert_eq!(result.accepted.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![key(2), key(1)]);
        assert_eq!(result.missing, vec![key(3)]);
        assert_eq!(guard.stats().discarded, 1);
        assert_eq!(guard.stats().bytes, 0);
    }

    #[test]
    fn stale_and_oversized_predictions_are_rejected() {
        let mut guard = ExpertCommitGuard::new(10, 1);
        guard.begin_token(5).unwrap();
        assert_eq!(guard.stage(5, key(1), block(9)), StageResult::Staged);
        assert_eq!(guard.stage(5, key(2), block(1)), StageResult::EntryLimit);
        assert_eq!(guard.stage(4, key(2), block(1)), StageResult::InvalidEpoch);
        guard.begin_token(6).unwrap();
        assert_eq!(guard.stage(6, key(2), block(11)), StageResult::ByteLimit);
        assert!(guard.commit(5, &[key(1)]).is_err());
        assert_eq!(guard.stats().bytes, 0);
    }

    #[test]
    fn duplicate_actual_route_fails_without_releasing_blocks() {
        let mut guard = ExpertCommitGuard::new(10, 1);
        guard.begin_token(1).unwrap();
        guard.stage(1, key(1), block(5));
        assert!(guard.commit(1, &[key(1), key(1)]).is_err());
        assert_eq!(guard.stats().accepted, 0);
    }

    #[test]
    fn shared_handle_survives_guard_reset() {
        let mut guard = ExpertCommitGuard::new(10, 1);
        guard.begin_token(1).unwrap();
        let handle = block(5);
        guard.stage(1, key(1), Arc::clone(&handle));
        let result = guard.commit(1, &[key(1)]).unwrap();
        assert!(Arc::ptr_eq(&handle, &result.accepted[0].1));
        guard.begin_token(2).unwrap();
        assert_eq!(result.accepted[0].1.as_ref(), &[3u8; 5]);
    }
}
