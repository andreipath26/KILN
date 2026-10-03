//! Functional test for the selector. Builds a mock manifest and envelope,
//! runs the selector, and prints the resulting plan.

use std::time::SystemTime;

use kiln_core::monitor::{MockMonitor, PerformanceEnvelope};
use kiln_core::selector::{DefaultSelector, ModeSelector};
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
    fn residency_state(&self, _tensor_id: u64) -> ResidencyState { ResidencyState::NotLoaded }
    fn transfer_latency(&self, _to: BackendId) -> u64 { 0 }
    fn synchronization_overhead(&self) -> u64 { 0 }
    fn thermal_state(&self) -> ThermalState { ThermalState::Warm }
}

fn main() {
    let mut registry = InMemoryRegistry::new();
    registry.register(Box::new(MockCpu));

    let manifest = ModelManifest {
        schema_version: 1,
        name: "test-model".to_string(),
        source_url: "local".to_string(),
        source_checksum: "deadbeef".to_string(),
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
            cold_start: Plan { name: "cold".to_string(), budgets: vec![TierBudget { tier: 0, max_ram_bytes: 12 * 1024 * 1024 * 1024, max_vram_bytes: 0, expected_tokens_per_sec: 4.0 }], algorithm_preference: vec!["Diffusion".to_string()] },
            steady_state: Plan { name: "steady".to_string(), budgets: vec![TierBudget { tier: 0, max_ram_bytes: 10 * 1024 * 1024 * 1024, max_vram_bytes: 0, expected_tokens_per_sec: 2.0 }], algorithm_preference: vec!["TernaryAr".to_string()] },
        },
        experts_index_path: None,
        experts_blob_path: None,
        kv_layout_path: None,
        weights_path: "weights.gguf".to_string(),
        ml_cache_dir: None,
    };

    let mut env = PerformanceEnvelope::baseline(12 * 1024 * 1024 * 1024);
    env.throughput_fraction = 1.0;
    env.thermal_state = ThermalState::Cold;
    env.sampled_at = SystemTime::now();

    let monitor = MockMonitor::new(env);

    // Test 1: full rung with short prompt
    let selector = DefaultSelector { num_layers: 32, prompt_tokens: 100 };
    let out = selector.select(&manifest, &env, &registry, &monitor);
    println!("=== TEST 1: full rung, short prompt ===");
    println!("rung:     {:?}", out.rung);
    println!("nodes:    {}", out.plan.nodes.len());
    println!("edges:    {}", out.plan.edges.len());
    println!("revision: {}", out.plan.revision);
    println!("first 3 reasons: {:?}", &out.reasons[..3.min(out.reasons.len())]);
    println!("first node: id={:?} op={:?} precision={:?} algorithm={:?}",
        out.plan.nodes[0].id, out.plan.nodes[0].op,
        out.plan.nodes[0].precision, out.plan.nodes[0].algorithm);
    println!("");

    // Test 2: full rung with long prompt (should prefer Diffusion)
    let selector2 = DefaultSelector { num_layers: 32, prompt_tokens: 1000 };
    let out2 = selector2.select(&manifest, &env, &registry, &monitor);
    println!("=== TEST 2: full rung, long prompt ===");
    println!("rung:     {:?}", out2.rung);
    println!("first reason: {:?}", out2.reasons[0]);
    println!("first node algorithm: {:?}", out2.plan.nodes[0].algorithm);
    println!("");

    // Test 3: low throughput, should drop to DisableSpecDecode or lower
    let mut env3 = env;
    env3.throughput_fraction = 0.55;
    let monitor3 = MockMonitor::new(env3);
    let out3 = selector.select(&manifest, &env3, &registry, &monitor3);
    println!("=== TEST 3: throughput 0.55 ===");
    println!("rung:     {:?}", out3.rung);
    println!("first node precision: {:?}", out3.plan.nodes[0].precision);
    println!("");

    // Test 4: severely degraded, should be Paused or SmallestModel
    let mut env4 = env;
    env4.throughput_fraction = 0.20;
    env4.thermal_state = ThermalState::SeverelyThrottling;
    let monitor4 = MockMonitor::new(env4);
    let out4 = selector.select(&manifest, &env4, &registry, &monitor4);
    println!("=== TEST 4: throughput 0.20, severely throttling ===");
    println!("rung:     {:?}", out4.rung);
    println!("is_paused: {}", out4.rung.is_paused());
    println!("");

    println!("=== END ===");
}
