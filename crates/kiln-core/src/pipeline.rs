//! The end-to-end inference pipeline.
//!
//! See docs/pipeline-design.md for the design. This is the minimal path
//! that loads a synthetic model, selects a plan, schedules it, executes
//! one matmul through the fused kernel, and reports the timing.

use std::path::PathBuf;
use std::time::Instant;

use kiln_kernels::matmul_scalar;
use kiln_models::manifest::{Architecture, ModelManifest};
use kiln_models::quantization::QuantizationSpec;
use kiln_models::thermal::{Plan, ThermalProfile, TierBudget};
use kiln_hal::registry::InMemoryRegistry;
use kiln_hal::Precision;

use crate::monitor::{MockMonitor, PerformanceEnvelope};
use crate::scheduler::{DefaultScheduler, Scheduler, StepOutcome};
use crate::selector::{DefaultSelector, ModeSelector};

/// Errors from the pipeline.
#[derive(Debug)]
pub enum PipelineError {
    EmptyModel,
    ZeroColumns,
    KernelFailed(&'static str),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::EmptyModel => write!(f, "synthetic model has zero rows"),
            PipelineError::ZeroColumns => write!(f, "synthetic model has zero columns"),
            PipelineError::KernelFailed(msg) => write!(f, "kernel failed: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Which loader to use for the model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loader {
    /// Real GGUF model file.
    Gguf,
}

impl Default for Loader {
    fn default() -> Self {
        Loader::Gguf
    }
}

/// A loaded model, regardless of source format. Provides the fields the
/// pipeline needs: packed TQ1_0 weight rows, activations, and a scale.
struct LoadedModel {
    num_rows: u32,
    num_cols: u32,
    scale: f32,
    weights: Vec<u8>,
    activations: Vec<i8>,
}

impl LoadedModel {
    fn total_weights(&self) -> usize {
        self.num_rows as usize * self.num_cols as usize
    }
}

fn load_gguf(path: &std::path::Path) -> Result<LoadedModel, PipelineError> {
    use kiln_models::gguf::GgufFile;
    let g = GgufFile::open(path).map_err(|e| {
        PipelineError::KernelFailed(Box::leak(format!("gguf: {}", e).into_boxed_str()))
    })?;

    // Find the first TQ1_0 tensor. That is the weight tensor.
    let weight_tensor = g.tensors.iter()
        .find(|t| matches!(t.dtype, kiln_models::gguf::GgufType::Tq1_0))
        .ok_or(PipelineError::KernelFailed("no TQ1_0 tensor found in GGUF"))?;

    if weight_tensor.shape.len() != 2 {
        return Err(PipelineError::KernelFailed(
            "weight tensor must be 2-dimensional"
        ));
    }
    let num_rows = weight_tensor.shape[0] as u32;
    let num_cols = weight_tensor.shape[1] as u32;

    let weight_bytes = g.tensor_bytes(&weight_tensor.name)
        .ok_or(PipelineError::KernelFailed("weight tensor bytes unavailable"))?
        .to_vec();

    // Find the first F32 tensor with shape [num_cols]. That is the
    // activation vector for our purpose. In a real model, activations
    // are computed, not stored. For pipeline testing, we look for a
    // tensor called "activations" or fall back to a synthetic vector.
    let acts: Vec<i8> = if let Some(t) = g.tensor("activations") {
        let bytes = g.tensor_bytes(&t.name)
            .ok_or(PipelineError::KernelFailed("activations bytes unavailable"))?;
        bytes.iter().map(|&b| b as i8).collect()
    } else {
        // No activations tensor. Use a fixed pattern so the pipeline
        // runs. A real model computes activations from input tokens.
        (0..num_cols).map(|i| ((i as i32 * 7 % 200) - 100) as i8).collect()
    };

    // Read the scale from metadata, or default to 1.0.
    let scale = g.metadata.get("kiln.scale")
        .and_then(|v| v.as_f32())
        .unwrap_or(1.0);

    Ok(LoadedModel {
        num_rows,
        num_cols,
        scale,
        weights: weight_bytes,
        activations: acts,
    })
}



/// The report produced by one pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub output: f32,
    pub load_time_nanos: u64,
    pub select_time_nanos: u64,
    pub schedule_time_nanos: u64,
    pub execute_time_nanos: u64,
    pub total_time_nanos: u64,
    pub nodes_executed: u32,
}

/// Run the pipeline once on a synthetic model file.
pub fn run_once(path: PathBuf, loader: Loader) -> Result<PipelineReport, PipelineError> {
    let total_start = Instant::now();

    // Step 1: Load.
    let load_start = Instant::now();
    let model = match loader {
        Loader::Gguf => load_gguf(&path)?,
    };
    if model.num_rows == 0 {
        return Err(PipelineError::EmptyModel);
    }
    if model.num_cols == 0 {
        return Err(PipelineError::ZeroColumns);
    }
    let _manifest = build_manifest(&model);
    let load_time = load_start.elapsed().as_nanos() as u64;

    // Step 2: Select. Build a plan from the manifest and a baseline
    // performance envelope.
    let select_start = Instant::now();
    let envelope = PerformanceEnvelope::baseline(12 * 1024 * 1024 * 1024);
    let registry = InMemoryRegistry::new();
    let monitor = MockMonitor::new(envelope);
    let selector = DefaultSelector {
        num_layers: model.num_rows,
        prompt_tokens: 0,
    };
    let outcome = selector.select(&_manifest, &envelope, &registry, &monitor);
    let plan = outcome.plan;
    let select_time = select_start.elapsed().as_nanos() as u64;

    // Step 3: Schedule. Create a scheduler with the plan.
    let schedule_start = Instant::now();
    let mut scheduler = DefaultScheduler::new(plan.clone(), 1.0);
    let _ = scheduler.revisions().len();
    let schedule_time = schedule_start.elapsed().as_nanos() as u64;

    // Step 4: Execute. Walk the plan and invoke the kernel for each node.
    // Each node corresponds to one output row.
    let execute_start = Instant::now();
    let mut first_output = 0.0f32;
    let stride = (model.num_cols as usize + 4) / 5;
    let mut nodes_executed = 0u32;

    while !scheduler.is_finished() {
        match scheduler.step() {
            Ok(StepOutcome::Executed { node, .. }) => {
                let r = node.0 as usize;
                let row_start = r * stride;
                let row_end = row_start + stride;
                if row_end > model.weights.len() {
                    return Err(PipelineError::KernelFailed("weight row out of bounds"));
                }
                let row_weights = &model.weights[row_start..row_end];
                let out = matmul_scalar(
                    row_weights,
                    &model.activations,
                    model.num_cols as usize,
                    model.scale,
                );
                if r == 0 {
                    first_output = out;
                }
                nodes_executed += 1;
            }
            Ok(StepOutcome::Finished) => break,
            Ok(StepOutcome::Revised { .. }) => continue,
            Err(e) => {
                return Err(PipelineError::KernelFailed(
                    if e.to_string().is_empty() { "scheduler error" } else { "scheduler error" }
                ));
            }
        }
    }
    let execute_time = execute_start.elapsed().as_nanos() as u64;

    let total_time = total_start.elapsed().as_nanos() as u64;

    Ok(PipelineReport {
        output: first_output,
        load_time_nanos: load_time,
        select_time_nanos: select_time,
        schedule_time_nanos: schedule_time,
        execute_time_nanos: execute_time,
        total_time_nanos: total_time,
        nodes_executed,
    })
}

/// Build a synthetic manifest from a synthetic model. In a real pipeline
/// this would come from the model's real metadata.
fn build_manifest(model: &LoadedModel) -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        name: "synthetic".to_string(),
        source_url: "local".to_string(),
        source_checksum: "none".to_string(),
        license: "MIT".to_string(),
        redistribution_allowed: true,
        architecture: Architecture::Dense,
        total_parameters: model.total_weights() as u64,
        active_parameters_per_token: 0,
        expert_layers: 0,
        experts_per_layer: 0,
        quantization: vec![QuantizationSpec {
            tensor_pattern: "*".to_string(),
            precision: Precision::Tq1_0,
            group_size: None,
            calibration: None,
        }],
        thermal_profile: ThermalProfile {
            cold_start: Plan {
                name: "cold".to_string(),
                budgets: vec![TierBudget {
                    tier: 0,
                    max_ram_bytes: 12 * 1024 * 1024 * 1024,
                    max_vram_bytes: 0,
                    expected_tokens_per_sec: 3.5,
                }],
                algorithm_preference: vec![],
            },
            steady_state: Plan {
                name: "steady".to_string(),
                budgets: vec![TierBudget {
                    tier: 0,
                    max_ram_bytes: 10 * 1024 * 1024 * 1024,
                    max_vram_bytes: 0,
                    expected_tokens_per_sec: 2.0,
                }],
                algorithm_preference: vec![],
            },
        },
        experts_index_path: None,
        experts_blob_path: None,
        kv_layout_path: None,
        weights_path: "synthetic".to_string(),
        ml_cache_dir: None,
    }
}

// Tests for the pipeline were removed when the synthetic model format
// was deleted. The pipeline is exercised end to end by the CLI:
//   kiln pipeline <gguf>
// Integration tests using a real GGUF file are a Phase 2 task. They
// require a small fixture GGUF checked into the repository. The current
// fixture is the real Qwen2.5 1.5B model at 986 MB, which is too large
// for routine tests.
