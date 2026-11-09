//! Phase 5.3: asynchronous speculation + exact-router Commit Guard.
use std::sync::Arc;
use kiln_mem::{ExpertCommitGuard, ExpertKey, CommitStats, StageResult, ExpertPrefetcher, PrefetchStats};
use crate::{CachedExpertBlobReader, AsyncExpertPrefetch, AsyncExpertStats, SubmitResult, ExpertWarmupReport};
use crate::{ExpertBlobError, RoutedExpertLoad};

/// A model session owns this coordinator. All predicted expert bytes originate
/// from the *same* verified source reader used by authoritative fallback.
pub struct AsyncGuardedExpertCoordinator {
    reader: Arc<CachedExpertBlobReader>,
    prefetch: AsyncExpertPrefetch,
    guard: ExpertCommitGuard,
    predictor: ExpertPrefetcher,
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
            predictor: ExpertPrefetcher::new(pending_requests, pending_bytes),
        }
    }

    /// Build both the verification/fallback reader and the native uring
    /// worker from the same validated expert file and index. The loader
    /// retains the exact-router Commit Guard for all completions.
    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_linux_uring(
        source: &std::path::Path,
        table: kiln_models::ExpertTable,
        cache_bytes: u64,
        cache_policy: kiln_mem::ExpertCachePolicy,
        pending_requests: usize,
        pending_bytes: u64,
        guard_bytes: u64,
        guard_entries: usize,
        try_sqpoll: bool,
    ) -> Result<Self, String> {
        let buffered=Arc::new(CachedExpertBlobReader::open(
            source, table.clone(), cache_bytes, cache_policy,
        ).map_err(|e| e.to_string())?);
        let native=crate::LinuxUringExpertReader::open(
            source, table, try_sqpoll,
        ).map_err(|e| e.to_string())?;
        let prefetch=AsyncExpertPrefetch::new_with_uring_reader(
            Arc::clone(&buffered),native,pending_requests,pending_bytes,
        );
        Ok(Self {
            reader:buffered,
            prefetch,
            guard:ExpertCommitGuard::new(guard_bytes,guard_entries),
            predictor: ExpertPrefetcher::new(pending_requests,pending_bytes),
        })
    }

    /// Explicitly prewarm verified expert weights according to a model-keyed
    /// frequency ranking. This never authorizes an expert for computation:
    /// commit still uses only the actual router's selected expert IDs.
    ///
    /// This call runs before a token epoch; it is synchronous and must not
    /// be described as overlapped I/O. The asynchronous prediction path is
    /// separate and continues to enforce epoch/byte limits.
    /// Bind an optional historical admission plan to the actual model's
    /// digest, checking every requested expert against the verified index.
    pub fn prewarm_history(
        &self,
        path: &std::path::Path,
        actual_model_sha256: &str,
    ) -> Result<ExpertWarmupReport, ExpertBlobError> {
        crate::prewarm_verified_history(&self.reader, path, actual_model_sha256)
    }

    pub fn prewarm_experts(
        &self,
        ranked: &[ExpertKey],
        max_bytes: u64,
    ) -> Result<ExpertWarmupReport, ExpertBlobError> {
        self.reader.prewarm_ranked(ranked, max_bytes)
    }

    /// Infer probable next expert weights from previously observed exact
    /// router choices, validating all proposed sizes against the source index.
    /// Only explicitly requested, bounded speculation is started; a predicted
    /// expert never replaces a different expert chosen by the true router.
    pub fn predict_from_router_trace(
        &mut self,
        epoch: u64,
        current: ExpertKey,
        candidates: &[ExpertKey],
        top_k: usize,
    ) -> Result<(usize, usize), ExpertBlobError> {
        if self.guard.epoch() != epoch || epoch == 0 {
            return Err(ExpertBlobError::InvalidRoute("prediction epoch mismatch".into()));
        }
        if candidates.len() > 4096 || top_k > 16 {
            return Err(ExpertBlobError::InvalidRoute(
                "unbounded expert prediction request".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let mut sizes = Vec::with_capacity(candidates.len());
        for &key in candidates {
            if seen.insert(key) {
                // Fail closed on an unindexed expert: do not let unverified
                // candidate sizes influence speculative I/O admission.
                let bytes = self.reader.expert_byte_length(key.layer, key.expert_id)?;
                sizes.push((key, bytes));
            }
        }
        let plans = self.predictor.predict(current, &sizes, top_k);
        let mut submitted = 0;
        let mut rejected = 0;
        for plan in plans {
            if self.prefetch.submit(epoch, plan.key) == SubmitResult::Submitted {
                submitted += 1;
            } else {
                self.predictor.cancel(plan.key);
                rejected += 1;
            }
        }
        Ok((submitted, rejected))
    }

    pub fn predictor_stats(&self) -> PrefetchStats { self.predictor.stats() }

    pub fn begin_epoch(&mut self, epoch: u64) -> Result<(), &'static str> {
        // Both layers reject non-increasing epochs. No completion from a
        // previously executed token can become an authorized expert.
        self.guard.begin_token(epoch)?;
        self.predictor.clear_pending();
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
                    self.predictor.mark_ready(done.key);
                    accepted += 1;
                } else {
                    self.predictor.cancel(done.key);
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
        // Real batched MoE graphs may route multiple token rows to one
        // physical expert. Never issue repeated disk reads or reject those
        // legal duplicate selections at the Commit Guard boundary.
        let mut unique = Vec::with_capacity(actual.len());
        let mut seen = std::collections::HashSet::new();
        for &key in actual {
            if seen.insert(key) {
                unique.push(key);
            }
        }
        let committed = self.guard.commit(epoch, &unique)
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
            let bytes = ready.get(&key).ok_or_else(|| {
                ExpertBlobError::InvalidRoute(
                    "authoritative expert block unavailable after commit".into(),
                )
            })?;
            experts.push((key, Arc::clone(bytes)));
        }
        for &key in actual {
            self.predictor.observe(key);
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
    fn advisory_warmup_reuses_verified_source_without_overriding_actual_router() {
        let root = std::env::temp_dir().join(format!(
            "kiln-verified-warmup-{}-{:?}",
            std::process::id(), std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("experts.bin");
        let data: Vec<u8> = (0..64u8).collect();
        std::fs::write(&source, &data).unwrap();
        let table = ExpertTable { blob_version: 1, blob_size: 64, ranges: vec![
            ExpertRange {layer: 0, expert_id: 0, byte_offset: 0, byte_length: 16},
            ExpertRange {layer: 0, expert_id: 1, byte_offset: 16, byte_length: 16},
            ExpertRange {layer: 0, expert_id: 2, byte_offset: 32, byte_length: 16},
        ]};
        let reader = Arc::new(CachedExpertBlobReader::open(
            &source, table, 32, ExpertCachePolicy::Lru
        ).unwrap());
        let mut coordinator = AsyncGuardedExpertCoordinator::new(
            Arc::clone(&reader), 2, 32, 32, 2
        );
        let key = |id| ExpertKey { layer: 0, expert_id: id };
        let warmed = coordinator.prewarm_experts(&[key(0), key(0), key(2)], 32).unwrap();
        assert_eq!(warmed.admitted, 2);
        assert_eq!(warmed.duplicate_keys, 1);
        assert_eq!(reader.cache_stats().hits, 0);
        coordinator.begin_epoch(1).unwrap();
        let routed = coordinator.load_actual(1, &[key(1), key(0)]).unwrap();
        assert_eq!(routed.predicted_hits, 0);
        assert_eq!(routed.authoritative_reads, 2);
        assert_eq!(routed.experts[0].0, key(1));
        assert_eq!(routed.experts[0].1.as_ref(), &data[16..32]);
        assert_eq!(routed.experts[1].0, key(0));
        assert_eq!(routed.experts[1].1.as_ref(), &data[0..16]);
        drop(routed);
        drop(coordinator);
        drop(reader);
        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn actual_batched_router_duplicates_share_one_verified_expert_allocation() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-duplicate-{}-{:?}",
            std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("source.bin");
        let bytes:Vec<u8>=(0..64u8).collect();
        std::fs::write(&source,&bytes).unwrap();
        let table=ExpertTable {blob_version:1,blob_size:64,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
            ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
        ]};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &source,table,32,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut coordinator=AsyncGuardedExpertCoordinator::new(
            reader,2,32,32,2,
        );
        let a=ExpertKey{layer:0,expert_id:0};
        let b=ExpertKey{layer:0,expert_id:1};
        coordinator.begin_epoch(1).unwrap();
        let result=coordinator.load_actual(1,&[a,b,a,b,a]).unwrap();
        assert_eq!(result.predicted_hits,0);
        assert_eq!(result.authoritative_reads,2);
        assert_eq!(result.experts.len(),5);
        assert_eq!(result.experts[0].1.as_ref(),&bytes[..16]);
        assert_eq!(result.experts[1].1.as_ref(),&bytes[16..32]);
        assert!(Arc::ptr_eq(&result.experts[0].1,&result.experts[2].1));
        assert!(Arc::ptr_eq(&result.experts[0].1,&result.experts[4].1));
        assert!(Arc::ptr_eq(&result.experts[1].1,&result.experts[3].1));
        drop(coordinator);
        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

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

    #[cfg(target_os = "linux")]
    #[test]
    fn ring_backed_prediction_is_verified_against_exact_router_selection() {
        let root=std::env::temp_dir().join(format!(
            "kiln-async-ring-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("expert.bin");
        let original:Vec<u8>=(0..96u8).collect();
        std::fs::write(&path,&original).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:96,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:32},
            ExpertRange{layer:0,expert_id:1,byte_offset:32,byte_length:32},
            ExpertRange{layer:0,expert_id:2,byte_offset:64,byte_length:32},
        ]};
        // Fall back to buffered reads automatically if io_uring setup is
        // unavailable under CI sandbox restrictions.
        let mut coordinator=AsyncGuardedExpertCoordinator::new_linux_uring(
            &path,table,96,ExpertCachePolicy::Lru,3,96,96,3,true,
        ).unwrap();
        let key=|id|ExpertKey{layer:0,expert_id:id};
        coordinator.begin_epoch(1).unwrap();
        assert_eq!(coordinator.schedule_prediction(1,key(0)),SubmitResult::Submitted);
        assert_eq!(coordinator.schedule_prediction(1,key(2)),SubmitResult::Submitted);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        while coordinator.prefetch_stats().completed<2 {
            coordinator.poll_and_stage();
            assert!(std::time::Instant::now()<deadline,"native worker timed out");
            std::thread::yield_now();
        }
        let result=coordinator.load_actual(1,&[key(1),key(0)]).unwrap();
        assert_eq!(result.predicted_hits,1);
        assert_eq!(result.authoritative_reads,1);
        assert_eq!(result.experts[0].1.as_ref(),&original[32..64]);
        assert_eq!(result.experts[1].1.as_ref(),&original[0..32]);
        assert_eq!(coordinator.guard_stats().bytes,0);
        drop(coordinator);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn live_router_history_predicts_only_verified_candidates_and_preserves_commit_guard() {
        let root=std::env::temp_dir().join(format!(
            "kiln-live-router-predict-{}-{:?}",
            std::process::id(), std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("experts.bin");
        let original:Vec<u8>=(0..64u8).collect();
        std::fs::write(&source,&original).unwrap();
        let table=ExpertTable {blob_version:1,blob_size:64,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
            ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
            ExpertRange{layer:0,expert_id:2,byte_offset:32,byte_length:16},
        ]};
        let reader=Arc::new(CachedExpertBlobReader::open(
            &source,table,48,ExpertCachePolicy::Lru,
        ).unwrap());
        let mut coordinator=AsyncGuardedExpertCoordinator::new(
            reader,2,32,32,2
        );
        let key=|id| ExpertKey {layer:0,expert_id:id};
        for epoch in 1..=3u64 {
            coordinator.begin_epoch(epoch).unwrap();
            let real=coordinator.load_actual(epoch,&[key(0),key(1)]).unwrap();
            assert_eq!(real.experts[1].1.as_ref(),&original[16..32]);
        }
        coordinator.begin_epoch(4).unwrap();
        assert!(coordinator.predict_from_router_trace(
            4,key(0),&[key(1),key(99)],1).is_err());
        assert_eq!(coordinator.predictor_stats().scheduled,0);
        let (submitted,rejected)=coordinator.predict_from_router_trace(
            4,key(0),&[key(1),key(2)],1
        ).unwrap();
        assert_eq!((submitted,rejected),(1,0));
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        while coordinator.prefetch_stats().completed<1 {
            coordinator.poll_and_stage();
            assert!(std::time::Instant::now()<deadline,
                "live router prefetch did not complete");
            std::thread::yield_now();
        }
        let result=coordinator.load_actual(4,&[key(2),key(1)]).unwrap();
        assert_eq!(result.predicted_hits,1);
        assert_eq!(result.authoritative_reads,1);
        assert_eq!(result.experts[0].0,key(2));
        assert_eq!(result.experts[0].1.as_ref(),&original[32..48]);
        assert_eq!(result.experts[1].0,key(1));
        assert_eq!(result.experts[1].1.as_ref(),&original[16..32]);
        assert_eq!(coordinator.predictor_stats().useful,1);
        drop(result);
        drop(coordinator);
        std::fs::remove_file(source).unwrap();
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
