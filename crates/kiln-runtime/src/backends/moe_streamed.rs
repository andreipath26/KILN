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
            RuntimeOp::MoEBeginToken { .. }
                | RuntimeOp::MoEPrefetch { .. }
                | RuntimeOp::MoEPoll
                | RuntimeOp::MoECommit { .. }
        )
    }

    fn estimate(&self, op: &RuntimeOp) -> Option<u64> {
        self.supports(op).then_some(0)
    }

    fn execute(&mut self, op: &RuntimeOp) -> Result<RuntimeResult, RuntimeError> {
        match op {
            RuntimeOp::MoEBeginToken { epoch } => {
                self.loader.begin_epoch(*epoch)
                    .map_err(|e| RuntimeError::ExpertStream(e.into()))?;
                Ok(RuntimeResult::Unit)
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
    fn resident_llama_backend_never_claims_stream_ops() {
        let backend = LlamaCppBackend::new();
        assert!(!backend.supports(&RuntimeOp::MoEPoll));
        assert!(!backend.supports(&RuntimeOp::MoEBeginToken{epoch:1}));
    }
}
