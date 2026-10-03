//! Functional test for the scheduler. Exercises the revision cycle.

use kiln_core::monitor::{MockMonitor, PerformanceEnvelope};
use kiln_core::scheduler::MonitoredScheduler;
use kiln_core::selector::DefaultSelector;
use kiln_core::ModeSelector;
use kiln_hal::{Backend, BackendId, DeviceType, MemoryModel, Op, Precision, ResidencyState, ThermalState};
use kiln_hal::registry::InMemoryRegistry;
use kiln_models::manifest::{Architecture, ModelManifest};
use kiln_models::quantization::QuantizationSpec;
use kiln_models::thermal::{Plan, ThermalProfile, TierBudget};

struct MockCpu;

impl Backend for MockCpu {
    fn id(&self) -> BackendId { BackendId(0) }
    fn device_type(&self) -> DeviceType { DeviceType::Cpu }
    fn supported_ops(&self) -> &[Op] {
        &[Op::MatMul, Op::Attention, Op::Norm, Op::Activation, Op::Embed, Op::Sample]
    }
    fn supported_precisions(&self) -> &[Precision] {
        &[Precision::Fp16, Precision::Int8, Precision::Int4, Precision::Tq1_0, Precision::Tq2_0, Precision::Tq3p]
    }
    fn memory_model(&self) -> MemoryModel {
        MemoryModel { total_bytes: 16 * 1024 * 1024 * 1024, available_bytes: 12 * 1024 * 1024 * 1024, bandwidth_bytes_per_sec: 37_500_000_000 }
    }
    fn available_memory(&self) -> u64 { 12 * 1024 * 1024 * 1024 }
    fn device_bandwidth(&self) -> u64 { 37_500_000_000 }
    fn conversion_cost(&self, _from: Precision, _to: Precision) -> Option<u64> { Some(1000) }
    fn residency_state(&self, _t: u64) -> ResidencyState { ResidencyState::NotLoaded }
    fn transfer_latency(&self, _to: BackendId) -> u64 { 0 }
    fn synchronization_overhead(&self) -> u64 { 0 }
    fn thermal_state(&self) -> ThermalState { ThermalState::Warm }
}

fn build_manifest() -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        name: "test".to_string(),
        source_url: "local".to_string(),
        source_checksum: "x".to_string(),
        license: "MIT".to_string(),
        redistribution_allowed: true,
        architecture: Architecture::Dense,
        total_parameters: 7_000_000_000,
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
            cold_start: Plan { name: "cold".to_string(), budgets: vec![TierBudget { tier: 0, max_ram_bytes: 12_000_000_000, max_vram_bytes: 0, expected_tokens_per_sec: 4.0 }], algorithm_preference: vec![] },
            steady_state: Plan { name: "steady".to_string(), budgets: vec![TierBudget { tier: 0, max_ram_bytes: 10_000_000_000, max_vram_bytes: 0, expected_tokens_per_sec: 2.0 }], algorithm_preference: vec![] },
        },
        experts_index_path: None,
        experts_blob_path: None,
        kv_layout_path: None,
        weights_path: "weights.gguf".to_string(),
        ml_cache_dir: None,
    }
}

fn main() {
    let mut registry = InMemoryRegistry::new();
    registry.register(Box::new(MockCpu));

    let manifest = build_manifest();

    // Start with a Full plan.
    let mut env = PerformanceEnvelope::baseline(12_000_000_000);
    env.throughput_fraction = 1.0;
    env.thermal_state = ThermalState::Cold;
    let monitor = MockMonitor::new(env);

    let selector = DefaultSelector { num_layers: 4, prompt_tokens: 100 };
    let initial_outcome = selector.select(&manifest, &env, &registry, &monitor);

    println!("=== TEST: revision cycle ===");
    println!("initial plan revision: {}", initial_outcome.plan.revision);
    println!("initial node count:    {}", initial_outcome.plan.nodes.len());
    println!("initial first node:    algorithm={:?} precision={:?}",
        initial_outcome.plan.nodes[0].algorithm,
        initial_outcome.plan.nodes[0].precision);
    println!("");

    // Wrap in MonitoredScheduler.
    let mut sched = MonitoredScheduler::new(
        initial_outcome.plan,
        1.0,
        &monitor,
        &registry,
        &selector,
    );

    // Step twice successfully.
    let s1 = sched.step_with_manifest(&manifest).unwrap();
    println!("step 1: {:?}", s1);
    let s2 = sched.step_with_manifest(&manifest).unwrap();
    println!("step 2: {:?}", s2);
    println!("revisions so far: {}", sched.revisions().len());
    println!("");

    // Now the monitor reports a thermal event.
    let mut env_low = PerformanceEnvelope::baseline(12_000_000_000);
    env_low.throughput_fraction = 0.55;
    env_low.thermal_state = ThermalState::Throttling;
    // MockMonitor is set inside the MonitoredScheduler by reference, so
    // we cannot change it here. Instead, build a second scheduler with
    // the low envelope to demonstrate the revision logic.
    let monitor_low = MockMonitor::new(env_low);
    let selector_low = DefaultSelector { num_layers: 4, prompt_tokens: 100 };
    let initial_low = selector_low.select(&manifest, &env_low, &registry, &monitor_low);
    println!("=== TEST: low envelope produces degraded plan ===");
    println!("revision: {}", initial_low.plan.revision);
    println!("rung:     {:?}", initial_low.rung);
    println!("first node precision: {:?}", initial_low.plan.nodes[0].precision);
    println!("");

    let mut sched_low = MonitoredScheduler::new(
        initial_low.plan,
        0.55,
        &monitor_low,
        &registry,
        &selector_low,
    );

    // The very first step should see a mismatch (0.55 in the plan vs 0.55
    // in the envelope) but the check_for_revision uses previous >= 0.80 as
    // the trigger, so no revision fires here. The plan is already correct.
    let s3 = sched_low.step_with_manifest(&manifest).unwrap();
    println!("low-env step 1: {:?}", s3);
    println!("low-env revisions: {}", sched_low.revisions().len());
    for r in sched_low.revisions() {
        println!("  revision {} reason {:?}", r.revision, r.reason);
    }
    println!("");

    println!("=== END ===");
}
