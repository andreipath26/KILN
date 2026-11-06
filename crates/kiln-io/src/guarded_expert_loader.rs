//! Phase 5.2 integration of speculative expert staging with authoritative reads.
//!
//! Predictions are advisory only. Final bytes always correspond to the
//! actual router keys, in the original router order.
use std::sync::Arc;
use kiln_mem::{CommitStats, ExpertCommitGuard, ExpertKey, StageResult};
use crate::{CachedExpertBlobReader, ExpertBlobError};

pub struct RoutedExpertLoad {
    pub experts: Vec<(ExpertKey, Arc<[u8]>)>,
    pub predicted_hits: usize,
    pub authoritative_reads: usize,
}

/// Keep the loader session-scoped. Call begin_token for each new router step.
/// Cache bytes and speculative retained bytes have separate budgets.
pub struct GuardedExpertLoader {
    reader: CachedExpertBlobReader,
    guard: ExpertCommitGuard,
}

impl GuardedExpertLoader {
    pub fn new(
        reader: CachedExpertBlobReader,
        speculative_byte_budget: u64,
        max_speculative_experts: usize,
    ) -> Self {
        Self {
            reader,
            guard: ExpertCommitGuard::new(speculative_byte_budget, max_speculative_experts),
        }
    }

    pub fn begin_token(&mut self, token_epoch: u64) -> Result<(), &'static str> {
        self.guard.begin_token(token_epoch)
    }

    /// Stage one guessed expert, using a shared cached allocation.
    /// Invalid epochs are rejected *before* any expert I/O.
    pub fn stage_prediction(
        &mut self,
        token_epoch: u64,
        key: ExpertKey,
    ) -> Result<StageResult, ExpertBlobError> {
        if token_epoch != self.guard_epoch() {
            return Ok(StageResult::InvalidEpoch);
        }
        let size = self.reader.expert_byte_length(key.layer, key.expert_id)?;
        let admitted = self.guard.can_stage(token_epoch, key, size);
        if admitted != StageResult::Staged { return Ok(admitted); }
        let bytes = self.reader.read_expert_shared(key.layer, key.expert_id)?;
        Ok(self.guard.stage(token_epoch, key, bytes))
    }

    fn guard_epoch(&self) -> u64 {
        self.guard.epoch()
    }

    /// Commit actual model-router selection. Missing predictions are read
    /// from the validated source, never filled with substitute experts.
    pub fn load_actual_route(
        &mut self,
        token_epoch: u64,
        actual: &[ExpertKey],
    ) -> Result<RoutedExpertLoad, ExpertBlobError> {
        let committed = self.guard.commit(token_epoch, actual)
            .map_err(|msg| ExpertBlobError::InvalidRoute(msg.to_owned()))?;
        let predicted_hits = committed.accepted.len();
        let authoritative_reads = committed.missing.len();
        let mut available: std::collections::HashMap<ExpertKey, Arc<[u8]>> =
            committed.accepted.into_iter().collect();
        for key in committed.missing {
            let bytes = self.reader.read_expert_shared(key.layer, key.expert_id)?;
            available.insert(key, bytes);
        }
        let mut experts = Vec::with_capacity(actual.len());
        for key in actual {
            let bytes = available.remove(key)
                .ok_or_else(|| ExpertBlobError::InvalidRoute(
                    "committed route is missing authoritative expert".into(),
                ))?;
            experts.push((*key, bytes));
        }
        Ok(RoutedExpertLoad { experts, predicted_hits, authoritative_reads })
    }

    pub fn discard_predictions(&mut self) { self.guard.cancel(); }
    pub fn commit_stats(&self) -> CommitStats { self.guard.stats() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::{ExpertTable, ExpertRange};

    #[test]
    fn wrong_prediction_is_never_substituted_for_real_router_expert() {
        let root = std::env::temp_dir().join(format!(
            "kiln-guarded-loader-{}-{:?}",
            std::process::id(), std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("experts.bin");
        let bytes:Vec<u8> = (0..64u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let table = ExpertTable {
            blob_version:1, blob_size:64, ranges:vec![
                ExpertRange {layer:0,expert_id:0,byte_offset:0,byte_length:16},
                ExpertRange {layer:0,expert_id:1,byte_offset:16,byte_length:16},
                ExpertRange {layer:0,expert_id:2,byte_offset:32,byte_length:16},
            ],
        };
        let reader=CachedExpertBlobReader::open(
            &path,table,32,ExpertCachePolicy::Lru,
        ).unwrap();
        let mut loader=GuardedExpertLoader::new(reader,32,2);
        let a=ExpertKey{layer:0,expert_id:0};
        let b=ExpertKey{layer:0,expert_id:1};
        let wrong=ExpertKey{layer:0,expert_id:2};
        loader.begin_token(1).unwrap();
        assert_eq!(loader.stage_prediction(1,a).unwrap(),StageResult::Staged);
        assert_eq!(loader.stage_prediction(1,wrong).unwrap(),StageResult::Staged);
        let actual=loader.load_actual_route(1,&[b,a]).unwrap();
        assert_eq!(actual.predicted_hits,1);
        assert_eq!(actual.authoritative_reads,1);
        assert_eq!(actual.experts[0].0,b);
        assert_eq!(actual.experts[0].1.as_ref(),&bytes[16..32]);
        assert_eq!(actual.experts[1].0,a);
        assert_eq!(actual.experts[1].1.as_ref(),&bytes[0..16]);
        assert_eq!(loader.commit_stats().bytes,0);
        loader.begin_token(2).unwrap();
        assert_eq!(loader.stage_prediction(1,a).unwrap(),StageResult::InvalidEpoch);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&root).unwrap();
    }
}
