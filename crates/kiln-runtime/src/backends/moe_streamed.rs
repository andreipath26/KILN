//! Phase 5.4 dispatch seam: CPU streamed expert-byte operations.
//!
//! Deliberately does not pretend that llama.cpp model inference is streamed.
//! The actual MoE compute graph must call these operations at the router
//! boundary, and the final backend ship gate still requires Dell acceptance.
use crate::runtime_dispatch::{
    RuntimeBackend, RuntimeError, RuntimeOp, RuntimeResult,
};
use kiln_io::{AsyncGuardedExpertCoordinator, SubmitResult};

pub struct MoEStreamBackend {
    loader: AsyncGuardedExpertCoordinator,
}

impl MoEStreamBackend {
    pub fn new(loader: AsyncGuardedExpertCoordinator) -> Self {
        Self { loader }
    }

    /// Advisory, opt-in cache warmup for a model's verified expert index.
    /// The runtime dispatcher still only commits actual router keys.
    pub fn prewarm_experts(
        &self,
        ranked: &[kiln_mem::ExpertKey],
        max_bytes: u64,
    ) -> Result<kiln_io::ExpertWarmupReport, RuntimeError> {
        self.loader.prewarm_experts(ranked, max_bytes)
            .map_err(|error| RuntimeError::ExpertStream(error.to_string()))
    }
}

impl RuntimeBackend for MoEStreamBackend {
    fn name(&self) -> &str { "cpu_moe_streamed" }

    fn supports(&self, op: &RuntimeOp) -> bool {
        matches!(
            op,
            RuntimeOp::MoEWarmup { .. }
                | RuntimeOp::MoEWarmupHistory { .. }
                | RuntimeOp::MoEBeginToken { .. }
                | RuntimeOp::MoEPrefetch { .. }
                | RuntimeOp::MoETracePredict { .. }
                | RuntimeOp::MoEPoll
                | RuntimeOp::MoECommit { .. }
        )
    }

    fn estimate(&self, op: &RuntimeOp) -> Option<u64> {
        self.supports(op).then_some(0)
    }

    fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
        match op {
            RuntimeOp::MoEWarmup { experts, max_bytes } => {
                Ok(RuntimeResult::MoEWarmupReport(
                    self.prewarm_experts(experts, *max_bytes)?
                ))
            }
            RuntimeOp::MoEWarmupHistory { path, actual_model_sha256 } => {
                let report = self.loader.prewarm_history(path, actual_model_sha256)
                    .map_err(|error| RuntimeError::ExpertStream(error.to_string()))?;
                Ok(RuntimeResult::MoEWarmupReport(report))
            }
            RuntimeOp::MoEBeginToken { epoch } => {
                self.loader.begin_epoch(*epoch)
                    .map_err(|e| RuntimeError::ExpertStream(e.into()))?;
                Ok(RuntimeResult::Unit)
            }
            RuntimeOp::MoETracePredict { epoch, current, candidates, top_k } => {
                let (submitted, rejected) = self.loader
                    .predict_from_router_trace(*epoch, *current, candidates, *top_k)
                    .map_err(|e| RuntimeError::ExpertStream(e.to_string()))?;
                Ok(RuntimeResult::MoEPrefetchSubmitted { submitted, rejected })
            }
            RuntimeOp::MoEPrefetch { epoch, experts } => {
                let mut submitted = 0usize;
                let mut rejected = 0usize;
                for &key in experts {
                    match self.loader.schedule_prediction(*epoch, key) {
                        SubmitResult::Submitted => submitted += 1,
                        _ => rejected += 1,
                    }
                }
                Ok(RuntimeResult::MoEPrefetchSubmitted { submitted, rejected })
            }
            RuntimeOp::MoEPoll => {
                let count = self.loader.poll_and_stage();
                Ok(RuntimeResult::MoEStaged { count })
            }
            RuntimeOp::MoECommit { epoch, actual } => {
                // A batched real MoE router can select the same expert for
                // multiple token rows. Commit Guard correctly accounts for
                // distinct physical expert blobs only; reconstruct original
                // row order using shared Arcs without duplicate disk reads.
                let mut seen = std::collections::HashSet::new();
                let mut unique = Vec::with_capacity(actual.len());
                for &key in actual {
                    if seen.insert(key) {
                        unique.push(key);
                    }
                }
                let result = self.loader.load_actual(*epoch, &unique)
                    .map_err(|e| RuntimeError::ExpertStream(e.to_string()))?;
                let by_key: std::collections::HashMap<_, _> =
                    result.experts.into_iter().collect();
                let experts = actual.iter().map(|key| {
                    let data = by_key.get(key).ok_or_else(|| RuntimeError::ExpertStream(
                        "verified expert missing during actual-route reconstruction".into()
                    ))?;
                    Ok((*key, std::sync::Arc::clone(data)))
                }).collect::<Result<Vec<_>, RuntimeError>>()?;
                Ok(RuntimeResult::MoEExpertBytes {
                    experts,
                    predicted_hits: result.predicted_hits,
                    authoritative_reads: result.authoritative_reads,
                })
            }
            _ => Err(RuntimeError::NoBackend),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlamaCppBackend, RuntimeDispatcher};
    use std::sync::Arc;
    use kiln_io::CachedExpertBlobReader;
    use kiln_mem::{ExpertCachePolicy, ExpertKey};
    use kiln_models::{ExpertRange, ExpertTable};

    #[test]
    fn dispatcher_warmup_respects_budget_and_never_changes_router_decisions() {
        let root = std::env::temp_dir().join(format!(
            "kiln-dispatch-warmup-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("experts.bin");
        let bytes: Vec<u8> = (0..48u8).collect();
        std::fs::write(&source, &bytes).unwrap();
        let table = ExpertTable { blob_version: 1, blob_size: 48, ranges: vec![
            ExpertRange { layer: 0, expert_id: 0, byte_offset: 0, byte_length: 16 },
            ExpertRange { layer: 0, expert_id: 1, byte_offset: 16, byte_length: 16 },
            ExpertRange { layer: 0, expert_id: 2, byte_offset: 32, byte_length: 16 },
        ]};
        let reader = Arc::new(CachedExpertBlobReader::open(
            &source, table, 32, ExpertCachePolicy::Lru
        ).unwrap());
        let coordinator = AsyncGuardedExpertCoordinator::new(
            Arc::clone(&reader), 2, 32, 32, 2,
        );
        let mut dispatch = RuntimeDispatcher::new();
        dispatch.register(Box::new(LlamaCppBackend::new()));
        dispatch.register(Box::new(MoEStreamBackend::new(coordinator)));
        let key = |id| ExpertKey { layer: 0, expert_id: id };
        let result = dispatch.dispatch(RuntimeOp::MoEWarmup {
            experts: vec![key(0), key(0), key(2), key(1)],
            max_bytes: 32,
        }).unwrap();
        match result {
            RuntimeResult::MoEWarmupReport(report) => {
                assert_eq!(report.requested, 4);
                assert_eq!(report.duplicate_keys, 1);
                assert_eq!(report.admitted, 2);
                assert_eq!(report.skipped_budget, 1);
                assert_eq!(report.admitted_bytes, 32);
            }
            _ => panic!("warmup operation was not handled by streaming backend"),
        }
        assert_eq!(reader.cache_stats().used_bytes, 32);
        assert_eq!(reader.cache_stats().hits, 0);
        dispatch.dispatch(RuntimeOp::MoEBeginToken { epoch: 1 }).unwrap();
        let result = dispatch.dispatch(RuntimeOp::MoECommit {
            epoch: 1, actual: vec![key(1), key(0), key(1)]
        }).unwrap();
        match result {
            RuntimeResult::MoEExpertBytes {
                experts, predicted_hits, authoritative_reads
            } => {
                assert_eq!(predicted_hits, 0);
                assert_eq!(authoritative_reads, 2);
                assert_eq!(experts[0].0, key(1));
                assert_eq!(experts[0].1.as_ref(), &bytes[16..32]);
                assert_eq!(experts[1].0, key(0));
                assert_eq!(experts[1].1.as_ref(), &bytes[..16]);
                assert!(Arc::ptr_eq(&experts[0].1, &experts[2].1));
            }
            _ => panic!("actual router result missing"),
        }
        drop(dispatch);
        drop(reader);
        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn registry_preserves_real_router_order_and_rejects_bad_predictions() {
        let root=std::env::temp_dir().join(format!(
            "kiln-stream-dispatch-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("experts.bin");
        let bytes:Vec<u8>=(0..80u8).collect();
        std::fs::write(&source,&bytes).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:80,ranges:vec![
            ExpertRange{layer:1,expert_id:0,byte_offset:0,byte_length:20},
            ExpertRange{layer:1,expert_id:1,byte_offset:20,byte_length:20},
            ExpertRange{layer:1,expert_id:2,byte_offset:40,byte_length:20},
        ]};
        let cache=Arc::new(CachedExpertBlobReader::open(
            &source,table,60,ExpertCachePolicy::Lru,
        ).unwrap());
        let coordinator=AsyncGuardedExpertCoordinator::new(cache,3,60,60,3);
        let mut dispatch=RuntimeDispatcher::new();
        // Native resident backend first: it must not accidentally intercept
        // streaming expert operations intended for the registered adapter.
        dispatch.register(Box::new(LlamaCppBackend::new()));
        dispatch.register(Box::new(MoEStreamBackend::new(coordinator)));
        let key=|id|ExpertKey{layer:1,expert_id:id};
        assert!(matches!(dispatch.dispatch(RuntimeOp::MoEBeginToken{epoch:1}),
            Ok(RuntimeResult::Unit)));
        match dispatch.dispatch(RuntimeOp::MoEPrefetch{
            epoch:1,experts:vec![key(0),key(2)],
        }).unwrap() {
            RuntimeResult::MoEPrefetchSubmitted{submitted,rejected} => {
                assert_eq!(submitted,2);
                assert_eq!(rejected,0);
            }
            _ => panic!("wrong return type"),
        }
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(5);
        let mut staged=0usize;
        while staged<2 {
            match dispatch.dispatch(RuntimeOp::MoEPoll).unwrap() {
                RuntimeResult::MoEStaged{count}=>staged+=count,
                _=>panic!("wrong return type"),
            }
            assert!(std::time::Instant::now()<deadline,
                "background prefetch did not complete");
            std::thread::yield_now();
        }
        match dispatch.dispatch(RuntimeOp::MoECommit{
            epoch:1,actual:vec![key(1),key(0),key(1),key(0)],
        }).unwrap() {
            RuntimeResult::MoEExpertBytes{
                experts,predicted_hits,authoritative_reads,
            } => {
                assert_eq!(predicted_hits,1);
                assert_eq!(authoritative_reads,1);
                assert_eq!(experts[0].0,key(1));
                assert_eq!(experts[0].1.as_ref(),&bytes[20..40]);
                assert_eq!(experts[1].0,key(0));
                assert_eq!(experts[1].1.as_ref(),&bytes[0..20]);
                assert_eq!(experts.len(),4);
                assert_eq!(experts[2].0,key(1));
                assert_eq!(experts[3].0,key(0));
                assert!(Arc::ptr_eq(&experts[0].1,&experts[2].1));
                assert!(Arc::ptr_eq(&experts[1].1,&experts[3].1));
            }
            _ => panic!("wrong return type"),
        }
        assert!(matches!(
            dispatch.dispatch(RuntimeOp::MoECommit{
                epoch:0,actual:vec![key(2)],
            }),
            Err(RuntimeError::ExpertStream(_))
        ));
        drop(dispatch);
        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn trace_prediction_rejects_invalid_epochs_and_out_of_index_candidates() {
        let root = std::env::temp_dir().join(format!(
            "kiln-dispatch-trace-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("experts.bin");
        std::fs::write(&path, [3u8;32]).unwrap();
        let table = ExpertTable {blob_version:1,blob_size:32,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16},
            ExpertRange{layer:0,expert_id:1,byte_offset:16,byte_length:16},
        ]};
        let source = Arc::new(CachedExpertBlobReader::open(
            &path,table,32,ExpertCachePolicy::Lru
        ).unwrap());
        let coordinator = AsyncGuardedExpertCoordinator::new(
            Arc::clone(&source),2,32,32,2
        );
        let mut d = RuntimeDispatcher::new();
        d.register(Box::new(LlamaCppBackend::new()));
        d.register(Box::new(MoEStreamBackend::new(coordinator)));
        let key = |id| ExpertKey{layer:0,expert_id:id};
        assert!(matches!(d.dispatch(RuntimeOp::MoETracePredict {
            epoch:1,current:key(0),candidates:vec![key(1)],top_k:1,
        }),Err(RuntimeError::ExpertStream(_))));
        d.dispatch(RuntimeOp::MoEBeginToken{epoch:1}).unwrap();
        assert!(matches!(d.dispatch(RuntimeOp::MoETracePredict {
            epoch:1,current:key(0),candidates:vec![key(99)],top_k:1,
        }),Err(RuntimeError::ExpertStream(_))));
        match d.dispatch(RuntimeOp::MoETracePredict {
            epoch:1,current:key(0),candidates:vec![key(1)],top_k:1,
        }).unwrap() {
            RuntimeResult::MoEPrefetchSubmitted{submitted,rejected} => {
                assert_eq!((submitted,rejected),(0,0));
            }
            _=>panic!("unexpected prediction result"),
        }
        assert_eq!(source.cache_stats().misses,0);
        drop(d);
        drop(source);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn resident_llama_backend_never_claims_stream_ops() {
        let backend = LlamaCppBackend::new();
        assert!(!backend.supports(&RuntimeOp::MoEWarmup {
            experts: vec![], max_bytes: 8,
        }));
        assert!(!backend.supports(&RuntimeOp::MoEWarmupHistory {
            path: std::path::PathBuf::from("missing-history.json"),
            actual_model_sha256: "a".repeat(64),
        }));
        assert!(!backend.supports(&RuntimeOp::MoETracePredict {
            epoch: 1, current: ExpertKey { layer: 0, expert_id: 0 },
            candidates: vec![], top_k: 0,
        }));
        assert!(!backend.supports(&RuntimeOp::MoEPoll));
        assert!(!backend.supports(&RuntimeOp::MoEBeginToken{epoch:1}));
    }
}
