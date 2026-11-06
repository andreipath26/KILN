//! Phase 5.3: asynchronous speculation + exact-router Commit Guard.
use std::sync::Arc;
use kiln_mem::{ExpertCommitGuard, ExpertKey, CommitStats, StageResult};
use crate::{CachedExpertBlobReader, AsyncExpertPrefetch, AsyncExpertStats, SubmitResult};
use crate::{ExpertBlobError, RoutedExpertLoad};

/// A model session owns this coordinator. All predicted expert bytes originate
/// from the *same* verified source reader used by authoritative fallback.
pub struct AsyncGuardedExpertCoordinator {
    reader: Arc<CachedExpertBlobReader>,
    prefetch: AsyncExpertPrefetch,
    guard: ExpertCommitGuard,
}

impl AsyncGuardedExpertCoordinator {
    pub fn new(
        reader: Arc<CachedExpertBlobReader>,
        pending_requests: usize,
        pending_bytes: u64,
        speculative_guard_bytes: u64,
        speculative_guard_entries: usize,
    ) -> Self {
        let prefetch = AsyncExpertPrefetch::new(
            Arc::clone(&reader), pending_requests, pending_bytes,
        );
        Self {
            reader, prefetch,
            guard: ExpertCommitGuard::new(
                speculative_guard_bytes, speculative_guard_entries,
            ),
        }
    }

    pub fn begin_epoch(&mut self, epoch: u64) -> Result<(), &'static str> {
        // Both layers reject non-increasing epochs. No completion from a
        // previously executed token can become an authorized expert.
        self.guard.begin_token(epoch)?;
        self.prefetch.begin_epoch(epoch)
    }

    pub fn schedule_prediction(
        &mut self, epoch: u64, key: ExpertKey,
    ) -> SubmitResult {
        self.prefetch.submit(epoch, key)
    }

    /// Drain ready reads without blocking. Prediction errors and stale
    /// completions are not inserted into the Commit Guard.
    pub fn poll_and_stage(&mut self) -> usize {
        let mut accepted = 0usize;
        for done in self.prefetch.poll_ready() {
            if let Ok(bytes) = done.result {
                if self.guard.stage(done.epoch, done.key, bytes) == StageResult::Staged {
                    accepted += 1;
                }
            }
        }
        accepted
    }

    /// This is the only path that releases expert bytes for computation.
    /// The actual model router's keys—not the speculative proposal order—
    /// determine the returned blocks. Every missing expert is loaded from
    /// the original verified source, even if every prediction was wrong.
    pub fn load_actual(
        &mut self, epoch: u64, actual: &[ExpertKey],
    ) -> Result<RoutedExpertLoad, ExpertBlobError> {
        self.poll_and_stage();
        let committed = self.guard.commit(epoch, actual)
            .map_err(|err| ExpertBlobError::InvalidRoute(err.into()))?;
        let predicted_hits = committed.accepted.len();
        let authoritative_reads = committed.missing.len();
        let mut ready: std::collections::HashMap<ExpertKey, Arc<[u8]>> =
            committed.accepted.into_iter().collect();
        for key in committed.missing {
            ready.insert(key, self.reader.read_expert_shared(key.layer, key.expert_id)?);
        }
        let mut experts = Vec::with_capacity(actual.len());
        for &key in actual {
            let bytes = ready.remove(&key).ok_or_else(|| {
                ExpertBlobError::InvalidRoute(
                    "authoritative expert block unavailable after commit".into(),
                )
            })?;
            experts.push((key, bytes));
        }
        Ok(RoutedExpertLoad {
            experts, predicted_hits, authoritative_reads,
        })
    }

    pub fn guard_stats(&self) -> CommitStats { self.guard.stats() }
    pub fn prefetch_stats(&self) -> AsyncExpertStats { self.prefetch.stats() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_mem::ExpertCachePolicy;
    use kiln_models::{ExpertTable, ExpertRange};

    #[test]
    fn async_prefetch_cannot_override_actual_router_experts() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-commit-{}-{:?}",
            std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let file=root.join("experts.bin");
        let original:Vec<u8>=(0..64u8).collect();
        std::fs::write(&file,&original).unwrap();
        let table=ExpertTable {
            blob_version:1,blob_size:64,ranges:vec![
                ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
                ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
                ExpertRange{layer:0,expert_id:2,byte_offset:32,byte_length:16},
            ],
        };
        let reader=Arc::new(CachedExpertBlobReader::open(
            &file,table,48,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut coordinator=AsyncGuardedExpertCoordinator::new(
            reader,3,48,48,3,
        );
        let key=|id|ExpertKey{layer:0,expert_id:id};
        coordinator.begin_epoch(1).unwrap();
        assert_eq!(coordinator.schedule_prediction(1,key(0)),SubmitResult::Submitted);
        assert_eq!(coordinator.schedule_prediction(1,key(2)),SubmitResult::Submitted);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        while coordinator.prefetch_stats().completed<2 {
            coordinator.poll_and_stage();
            assert!(std::time::Instant::now()<deadline,"expert prefetch timed out");
            std::thread::yield_now();
        }
        let actual=coordinator.load_actual(1,&[key(1),key(0)]).unwrap();
        assert_eq!(actual.predicted_hits,1);
        assert_eq!(actual.authoritative_reads,1);
        assert_eq!(actual.experts[0].0,key(1));
        assert_eq!(actual.experts[0].1.as_ref(),&original[16..32]);
        assert_eq!(actual.experts[1].0,key(0));
        assert_eq!(actual.experts[1].1.as_ref(),&original[0..16]);
        assert_eq!(coordinator.guard_stats().bytes,0);
        coordinator.begin_epoch(2).unwrap();
        assert_eq!(coordinator.schedule_prediction(1,key(0)),SubmitResult::StaleEpoch);
        drop(coordinator);
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn incorrect_router_epoch_releases_no_predicted_expert() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-epoch-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let file=root.join("experts.bin");
        std::fs::write(&file,[7u8;16]).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:16,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
        ]};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &file,table,16,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut coordinator=AsyncGuardedExpertCoordinator::new(
            reader,1,16,16,1,
        );
        coordinator.begin_epoch(1).unwrap();
        let key=ExpertKey{layer:0,expert_id:0};
        assert!(coordinator.load_actual(2,&[key]).is_err());
        assert_eq!(coordinator.guard_stats().accepted,0);
        drop(coordinator);
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
