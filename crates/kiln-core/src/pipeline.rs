//! The end-to-end inference pipeline.
//!
//! See docs/pipeline-design.md for the design. This is the minimal path
//! that loads a synthetic model, selects a plan, schedules it, executes
//! one matmul through the fused kernel, and reports the timing.

use std::path::PathBuf;
use std::time::Instant;

use kiln_kernels::matmul_scalar;
use kiln_models::synthetic::{read_synthetic, SyntheticError, SyntheticModel};
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
    Synthetic(SyntheticError),
    EmptyModel,
    ZeroColumns,
    KernelFailed(&'static str),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Synthetic(e) => write!(f, "synthetic model error: {}", e),
            PipelineError::EmptyModel => write!(f, "synthetic model has zero rows"),
            PipelineError::ZeroColumns => write!(f, "synthetic model has zero columns"),
            PipelineError::KernelFailed(msg) => write!(f, "kernel failed: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<SyntheticError> for PipelineError {
    fn from(e: SyntheticError) -> Self {
        PipelineError::Synthetic(e)
    }
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
pub fn run_once(path: PathBuf) -> Result<PipelineReport, PipelineError> {
    let total_start = Instant::now();

    // Step 1: Load.
    let load_start = Instant::now();
    let model = read_synthetic(&path)?;
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
fn build_manifest(model: &SyntheticModel) -> ModelManifest {
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

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_models::synthetic::{write_synthetic, SyntheticModel};

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kiln_pipeline_test_{}_{}.bin", name, std::process::id()));
        p
    }

    #[test]
    fn pipeline_runs_end_to_end() {
        // 4 rows x 10 cols. All zero weights, activations 1..10.
        // Expected output for row 0: 0.0 (all trits are 0).
        let model = SyntheticModel {
            num_rows: 4,
            num_cols: 10,
            scale: 1.0,
            weights: vec![121u8; 8], // 121 = all digits are 1 = all trits 0
            activations: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };
        let path = tmp_path("e2e");
        write_synthetic(&path, &model).unwrap();
        let report = run_once(path.clone()).unwrap();
        assert_eq!(report.output, 0.0);
        assert_eq!(report.nodes_executed, 4);
        assert!(report.load_time_nanos > 0);
        assert!(report.execute_time_nanos > 0);
        assert!(report.total_time_nanos > 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pipeline_handles_all_ones() {
        // All trits +1. Expected row 0 output = sum of activations
        // = 1+2+...+10 = 55.
        let model = SyntheticModel {
            num_rows: 1,
            num_cols: 10,
            scale: 1.0,
            weights: vec![242u8; 2], // 242 = all digits are 2 = all trits +1
            activations: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };
        let path = tmp_path("ones");
        write_synthetic(&path, &model).unwrap();
        let report = run_once(path.clone()).unwrap();
        assert_eq!(report.output, 55.0);
        std::fs::remove_file(&path).ok();
    }
}
