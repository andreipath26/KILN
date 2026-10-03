//! The Constraint-Aware Registry.
//!
//! The Registry aggregates every registered Backend and answers questions for
//! the scheduler. It does not make dispatch decisions. It provides the data
//! the scheduler needs to make them.

use crate::backend::{Backend, ConversionPath};
use crate::types::{BackendId, Op, Precision, ThermalState};

/// The Registry answers capability and cost questions about all backends.
pub trait Registry: Send + Sync {
    /// All registered backends.
    fn backends(&self) -> &[Box<dyn Backend>];

    /// Every backend that supports the given op at the given precision.
    fn backends_supporting(&self, op: Op, precision: Precision) -> Vec<BackendId>;

    /// The backend with the lowest estimated cost for the given op and
    /// precision under the current thermal state. Returns None if no backend
    /// supports the combination.
    fn fastest_backend_for(
        &self,
        op: Op,
        precision: Precision,
        thermal: ThermalState,
    ) -> Option<BackendId>;

    /// The full conversion path between two precisions across two backends.
    fn conversion_path(
        &self,
        from: Precision,
        to: Precision,
        from_backend: BackendId,
        to_backend: BackendId,
    ) -> Option<ConversionPath>;
}

/// A simple in-memory registry. Real implementations live in kiln-core.
pub struct InMemoryRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl InMemoryRegistry {
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }
}

impl Default for InMemoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry for InMemoryRegistry {
    fn backends(&self) -> &[Box<dyn Backend>] {
        &self.backends
    }

    fn backends_supporting(&self, op: Op, precision: Precision) -> Vec<BackendId> {
        self.backends
            .iter()
            .filter(|b| b.supported_ops().contains(&op))
            .filter(|b| b.supported_precisions().contains(&precision))
            .map(|b| b.id())
            .collect()
    }

    fn fastest_backend_for(
        &self,
        op: Op,
        precision: Precision,
        _thermal: ThermalState,
    ) -> Option<BackendId> {
        self.backends
            .iter()
            .filter(|b| b.supported_ops().contains(&op))
            .filter(|b| b.supported_precisions().contains(&precision))
            .max_by_key(|b| b.device_bandwidth())
            .map(|b| b.id())
    }

    fn conversion_path(
        &self,
        from: Precision,
        to: Precision,
        from_backend: BackendId,
        to_backend: BackendId,
    ) -> Option<ConversionPath> {
        if from == to {
            return Some(ConversionPath { steps: vec![], total_cost_nanos: 0 });
        }

        let src = self.backends.iter().find(|b| b.id() == from_backend)?;
        let dst = self.backends.iter().find(|b| b.id() == to_backend)?;

        // Case 1: source backend can perform the conversion directly.
        if let Some(cost) = src.conversion_cost(from, to) {
            return Some(ConversionPath {
                steps: vec![crate::backend::ConversionStep {
                    backend: from_backend,
                    from,
                    to,
                    cost_nanos: cost,
                }],
                total_cost_nanos: cost,
            });
        }

        // Case 2: destination backend can perform the conversion directly.
        if let Some(cost) = dst.conversion_cost(from, to) {
            return Some(ConversionPath {
                steps: vec![crate::backend::ConversionStep {
                    backend: to_backend,
                    from,
                    to,
                    cost_nanos: cost,
                }],
                total_cost_nanos: cost,
            });
        }

        // Case 3: route through FP16 on the source backend, then convert on
        // the destination. This is the fallback path. A future revision will
        // build a full graph shortest-path search across all backends.
        let hop1 = src.conversion_cost(from, Precision::Fp16)?;
        let hop2 = dst.conversion_cost(Precision::Fp16, to)?;
        let transfer = src.transfer_latency(to_backend);
        let total = hop1 + hop2 + transfer;
        Some(ConversionPath {
            steps: vec![
                crate::backend::ConversionStep {
                    backend: from_backend,
                    from,
                    to: Precision::Fp16,
                    cost_nanos: hop1,
                },
                crate::backend::ConversionStep {
                    backend: to_backend,
                    from: Precision::Fp16,
                    to,
                    cost_nanos: hop2,
                },
            ],
            total_cost_nanos: total,
        })
    }
}
