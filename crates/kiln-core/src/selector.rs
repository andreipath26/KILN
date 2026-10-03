//! The Automatic Mode Selector.
//!
//! Reads the model manifest, the current performance envelope, and the
//! Constraint-Aware Registry, then produces an ExecutionPlan. See
//! docs/core-design.md, Subsystem 2.
//!
//! Version 1 produces a linear chain of nodes, one per layer. MoE subgraphs
//! and diffusion loops land in Phase 1.

use std::time::SystemTime;

use kiln_hal::{Algorithm, BackendId, Op, Precision, Registry};
use kiln_models::{Architecture, ModelManifest};

use crate::monitor::{PerformanceEnvelope, PerformanceMonitor};
use crate::plan::{Edge, ExecutionPlan, Node, NodeId, TensorRef};

/// The degradation ladder rung. Higher is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LadderRung {
    /// Below 25 percent of baseline. Pause and notify.
    Paused = 0,
    /// 25 to 40 percent. Smallest model only.
    SmallestModel = 1,
    /// 40 to 60 percent. Drop KV cache to q4_0.
    DropKvPrecision = 2,
    /// 60 to 80 percent. Disable speculative decoding.
    DisableSpecDecode = 3,
    /// 80 to 100 percent. No change.
    Full = 4,
}

impl LadderRung {
    /// Pick the rung for the given throughput fraction.
    pub fn for_throughput(fraction: f32) -> Self {
        if fraction >= 0.80 {
            LadderRung::Full
        } else if fraction >= 0.60 {
            LadderRung::DisableSpecDecode
        } else if fraction >= 0.40 {
            LadderRung::DropKvPrecision
        } else if fraction >= 0.25 {
            LadderRung::SmallestModel
        } else {
            LadderRung::Paused
        }
    }

    /// Whether this rung pauses execution.
    pub fn is_paused(&self) -> bool {
        matches!(self, LadderRung::Paused)
    }
}

/// Why a particular algorithm was chosen for a node. Used for logging and
/// for the change log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmReason {
    /// Dense model on a GPU tier. Use Dense.
    DenseOnGpu,
    /// Dense model on a CPU-only tier with a long enough prompt. Use Diffusion.
    DenseOnCpuLongPrompt,
    /// Dense model on a CPU-only tier with a short prompt. Use TernaryAr.
    DenseOnCpuShortPrompt,
    /// MoE model. Use MoEStream for expert layers.
    MoeStream,
    /// SSM model. Use Dense.
    SsmDense,
    /// Fallback when no better algorithm is available.
    Fallback,
}

/// The result of one selection pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionOutcome {
    pub plan: ExecutionPlan,
    pub rung: LadderRung,
    pub reasons: Vec<AlgorithmReason>,
}

/// The Mode Selector trait.
pub trait ModeSelector: Send + Sync {
    /// Produce a plan from the given inputs.
    fn select(
        &self,
        manifest: &ModelManifest,
        envelope: &PerformanceEnvelope,
        registry: &dyn Registry,
        monitor: &dyn PerformanceMonitor,
    ) -> SelectionOutcome;
}

/// The default selector. Implements the five-step algorithm from the design
/// document.
pub struct DefaultSelector {
    /// Estimated number of layers in the model. In a real implementation
    /// this comes from the manifest. Kept as a constant for version 1.
    pub num_layers: u32,
    /// Prompt token count. Used to decide diffusion vs TernaryAr on CPU.
    pub prompt_tokens: u32,
}

impl Default for DefaultSelector {
    fn default() -> Self {
        Self {
            num_layers: 32,
            prompt_tokens: 0,
        }
    }
}

/// The prompt token count above which diffusion is preferred on CPU.
/// From Phase 0A Research Item 1.
pub const DIFFUSION_PROMPT_THRESHOLD: u32 = 256;

impl DefaultSelector {
    /// Choose the algorithm and reason for a single layer.
    fn choose_algorithm(
        &self,
        architecture: Architecture,
        is_attention: bool,
        has_gpu: bool,
    ) -> (Algorithm, AlgorithmReason) {
        match architecture {
            Architecture::Ssm => (Algorithm::Dense, AlgorithmReason::SsmDense),
            Architecture::Moe => {
                if is_attention {
                    (Algorithm::Dense, AlgorithmReason::DenseOnGpu)
                } else {
                    (Algorithm::MoeStream, AlgorithmReason::MoeStream)
                }
            }
            Architecture::Hybrid => {
                if is_attention {
                    (Algorithm::Dense, AlgorithmReason::DenseOnGpu)
                } else {
                    (Algorithm::TernaryAr, AlgorithmReason::DenseOnCpuLongPrompt)
                }
            }
            Architecture::Dense => {
                if has_gpu {
                    (Algorithm::Dense, AlgorithmReason::DenseOnGpu)
                } else if self.prompt_tokens >= DIFFUSION_PROMPT_THRESHOLD {
                    (Algorithm::Diffusion, AlgorithmReason::DenseOnCpuLongPrompt)
                } else {
                    (Algorithm::TernaryAr, AlgorithmReason::DenseOnCpuShortPrompt)
                }
            }
        }
    }

    /// Choose a precision for a layer given the rung.
    fn choose_precision(&self, rung: LadderRung, is_attention: bool) -> Precision {
        match rung {
            LadderRung::Full => {
                if is_attention {
                    Precision::Fp16
                } else {
                    Precision::Tq1_0
                }
            }
            LadderRung::DisableSpecDecode => Precision::Tq1_0,
            LadderRung::DropKvPrecision => Precision::Tq2_0,
            LadderRung::SmallestModel => Precision::Tq3p,
            LadderRung::Paused => Precision::Tq3p,
        }
    }

    /// Pick the placement backend for a layer.
    fn choose_placement(
        &self,
        registry: &dyn Registry,
        op: Op,
        precision: Precision,
        envelope: &PerformanceEnvelope,
    ) -> Option<BackendId> {
        registry.fastest_backend_for(op, precision, envelope.thermal_state)
    }
}

impl ModeSelector for DefaultSelector {
    fn select(
        &self,
        manifest: &ModelManifest,
        envelope: &PerformanceEnvelope,
        registry: &dyn Registry,
        _monitor: &dyn PerformanceMonitor,
    ) -> SelectionOutcome {
        // Step 1: pick the rung from the envelope.
        let rung = LadderRung::for_throughput(envelope.throughput_fraction);

        // Step 2: check if there is a GPU in the registry.
        let has_gpu = registry
            .backends()
            .iter()
            .any(|b| !matches!(b.device_type(), kiln_hal::DeviceType::Cpu));

        // Step 3: build nodes and reasons.
        let mut nodes = Vec::with_capacity(self.num_layers as usize);
        let mut edges = Vec::with_capacity(self.num_layers.saturating_sub(1) as usize);
        let mut reasons = Vec::with_capacity(self.num_layers as usize);
        let mut total_latency = 0u64;

        for layer_idx in 0..self.num_layers {
            // Version 1 assumes every layer is a dense transformer block
            // with one attention op and one FFN op. In a real implementation
            // the layer types come from the manifest.
            let is_attention = layer_idx % 2 == 0;

            let (algorithm, reason) =
                self.choose_algorithm(manifest.architecture, is_attention, has_gpu);
            let precision = self.choose_precision(rung, is_attention);

            let op = if is_attention { Op::Attention } else { Op::MatMul };

            let placement = self
                .choose_placement(registry, op, precision, envelope)
                .unwrap_or(BackendId(0));

            let estimated_cost_nanos = 1_000_000; // placeholder: 1 ms
            total_latency = total_latency.saturating_add(estimated_cost_nanos);

            let node = Node {
                id: NodeId(layer_idx),
                op,
                precision,
                placement,
                algorithm,
                estimated_cost_nanos,
                estimated_memory_bytes: 0,
                inputs: Vec::new(),
                outputs: Vec::new(),
            };
            nodes.push(node.clone());
            reasons.push(reason);

            if layer_idx > 0 {
                edges.push(Edge {
                    from: NodeId(layer_idx - 1),
                    to: NodeId(layer_idx),
                    tensor: TensorRef {
                        tensor: layer_idx as u64,
                        precision,
                        residency: kiln_hal::ResidencyState::NotLoaded,
                        byte_size: 0,
                    },
                    transfer_backend: placement,
                    transfer_cost_nanos: 0,
                });
            }
        }

        let plan = ExecutionPlan {
            revision: 1,
            nodes,
            edges,
            total_estimated_latency_nanos: total_latency,
            total_estimated_memory_bytes: 0,
            created_at: SystemTime::now(),
        };

        SelectionOutcome { plan, rung, reasons }
    }
}
