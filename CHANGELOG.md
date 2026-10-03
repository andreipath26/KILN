# Changelog

All notable changes to KILN are documented here.

The format is based on Keep a Changelog.

## [Unreleased]

### Phase 0B — Foundation (2026-10-03)

Thirteen commits. All eight crates created. cargo check passes with zero
warnings.

#### Added

- Repository skeleton with crates, cpp, backends, docs, models, tests,
  scripts folders.
- docs/architecture.md — six-layer runtime stack specification.
- docs/core-design.md — Performance Monitor, Mode Selector, DAG Scheduler.
- phase-0a-findings.md — seven confirmed research items, one conditional.
- KILN MASTER ROADMAP AND RULES SET.md — full roadmap and rulebook.
- kiln-hal crate: Backend trait, Registry trait, DeviceType, Precision,
  ResidencyState, ThermalState, Op, Algorithm, MemoryModel, ConversionCost,
  ConversionPath, ConversionStep, InMemoryRegistry.
- kiln-models crate: ModelManifest, Architecture, QuantizationSpec,
  ExpertTable, ExpertRange, KvLayout, KvExtent, ThermalProfile, Plan,
  TierBudget, MlCacheRef.
- kiln-mem crate: Tier (L3, Ram, Ssd, Vram), MemoryManager trait,
  InMemoryManager, RebalancePolicy, RebalanceReport, LoadError, EvictError,
  TensorId.
- kiln-io crate: IoOp, IoRequest, IoResult, IoSource trait, BlockSize,
  SyncIoSource fallback.
- kiln-core crate: PerformanceEnvelope, Sample, ThermalHistory,
  ThrottlePrediction, PerformanceMonitor trait, LinuxMonitor, MockMonitor,
  LadderRung, AlgorithmReason, SelectionOutcome, ModeSelector trait,
  DefaultSelector, SchedulerError, ExecutionPlan, Node, Edge, TensorRef,
  NodeId, PlanRevision, RevisionReason, Scheduler trait, DefaultScheduler,
  MonitoredScheduler, StepOutcome.
- kiln-api crate: GenerateRequest, GenerateResponse, ChatRequest,
  ChatMessage, ChatResponse, TagsResponse, ModelInfo, PullRequest,
  VersionResponse, four route handlers, axum router, serve function.
- kiln-cli crate: seven commands (serve, run, pull, list, bench, plan,
  info) with clap argument parsing. serve and info functional. Others exit
  cleanly with code 2 and a not-yet-implemented message.
- kiln-bench crate: BenchReport, Measurement, Summary, five Phase 0B
  benchmarks, human and JSON output modes.
- examples/monitor_test.rs, examples/selector_test.rs,
  examples/scheduler_test.rs, examples/server_test.rs.
- .gitignore, LICENSE (MIT).

#### Verified

- Monitor on Dell Latitude 7490: RAM 15.50 GB, CPU 2356 MHz, 70.1 C,
  thermal state Warm. Later run showed Throttling at 0.170 throughput
  after sustained cargo compilation. Live demonstration of Pillar 6.
- Selector: four test cases pass. Full rung short prompt uses TernaryAr.
  Full rung long prompt uses Diffusion. Throughput 0.55 uses
  DropKvPrecision. Throughput 0.20 uses Paused.
- Scheduler: revision cycle works. Initial plan revision 1, 4 nodes.
  Steps execute cleanly. Low envelope produces DropKvPrecision rung and
  Tq2_0 precision.
- API: all four endpoints respond correctly. GET /api/version returns
  KILN version. GET /api/tags returns empty model list. POST
  /api/generate and POST /api/chat return placeholder responses.
- CLI: kiln --help shows all seven commands. kiln info prints hardware
  profile. kiln list and kiln plan exit cleanly with code 2.
- Bench: five measurements recorded. monitor_envelope_read 248-283 ns.
  monitor_history_read 1777-1923 ns. monitor_predict 315-323 ns.

#### Known issues

- monitor_sample re-parses /proc/cpuinfo fully on every call. Takes
  99-185 ms. Optimization to cache parsed structure deferred to Phase 1.
- throughput_fraction baseline is fragile. Uses first-sample mean.
  Proper calibration deferred to Phase 1.
- pSLC write endurance tracking not yet implemented. Phase 3.
- Item 8 (thermal envelope measurement) requires AC-powered 30-minute
  test before Phase 2 gate.

### Phase 0A — Research Closure (2026-10-03)

Eight research items. Seven confirmed. One conditional.

- Item 1: Diffusion on 4-core CPU. CONFIRMED. Roofline paper validates
  the arithmetic intensity argument. 3-6 tok/s on 4 cores for easy
  prompts.
- Item 2: Ternary model quality. CONFIRMED WITH CAVEAT. Ternary Bonsai
  8B at 75.5 vs FP16 Qwen3-8B at 79.3. Knowledge recall caveat unresolved.
- Item 3: MoE locality. CONFIRMED. Cache hit rate is bottleneck.
  FlashMoE plus TIDE required.
- Item 4: KV cache quantization. CONFIRMED. q8_0 sufficient. HERALD
  and KVDRIVE for diffusion.
- Item 5: BitNet on AVX2. CONFIRMED. T-MAC and AYOT. Maple Preview
  20B-A1B at 28-34 tok/s on i5-8350U.
- Item 6: Thermal throttling. CONFIRMED. 90 C under load. 30 percent
  sustained. Pillar 6 required.
- Item 7: SSM vs transformer. CONFIRMED. BitMamba-2-1B at 52.86 tok/s
  on i3-12100F. Dense SSM viable. MoE SSM fails.
- Item 8: Thermal envelope. OPEN-CONDITIONAL. Requires AC-powered
  30-minute test.

## [Released]

None yet.
