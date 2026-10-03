# KILN Core Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for kiln-core. Anchors the DAG scheduler, the
Automatic Mode Selector, and the Performance Monitor.

## Non-goals

This document does not specify kernel implementations. It does not specify
backend details. It specifies the three core abstractions, their interfaces,
their state, and their interaction.

## Overview

kiln-core is the heart of the runtime. It contains three subsystems that
sit between the User API (Layer 6) and the Constraint-Aware Registry
(Layer 3) defined in docs/architecture.md.

  1. The Performance Monitor watches the machine.
  2. The Automatic Mode Selector decides what to run.
  3. The DAG Scheduler assembles and executes the plan.

The three interact in a cycle:

  Monitor observes machine
  Selector reads monitor + model metadata, produces a plan
  Scheduler executes the plan
  Monitor observes the result
  Selector revises the plan if the envelope changed
  Scheduler executes the revision

## Subsystem 1 — Performance Monitor

The Performance Monitor runs on a dedicated thread. It samples the machine
every 500 milliseconds and writes to a lock-free ring buffer. The scheduler
and selector read from the buffer without blocking the monitor.

What it samples:

  - CPU frequency per core, from /proc/cpuinfo on Linux
  - CPU package temperature, from /sys/class/thermal
  - Per-backend throughput over a rolling 10-second window
  - RAM pressure, from /proc/meminfo
  - SSD read and write throughput, from /proc/diskstats
  - Battery state, from /sys/class/power_supply

What it exposes:

  current_envelope() -> PerformanceEnvelope
  thermal_history(window: Duration) -> ThermalHistory
  predicted_throttle() -> Option<ThrottlePrediction>
  register_callback(cb) -> nothing

PerformanceEnvelope has three fields:

  throughput_fraction: f32, current throughput divided by cold-start baseline
  thermal_state: ThermalState, from kiln-hal
  ram_headroom_bytes: u64, bytes available before swap activity begins

ThrottlePrediction is Some when the thermal history suggests throttling
within the next 60 seconds. The prediction is based on a simple linear
extrapolation of the last 5 minutes of temperature and frequency data.
This is deliberately conservative. It errs on the side of predicting
throttling that does not occur rather than missing throttling that does.

The monitor samples in a fixed order. It reads /proc files directly rather
than shelling out. When a file is missing, the corresponding sample is
recorded as None and the envelope field is omitted.

## Subsystem 2 — Automatic Mode Selector

The selector runs at model load and again whenever the monitor reports a
significant event. A significant event is either:

  - thermal state changed to Throttling or SeverelyThrottling
  - throughput_fraction dropped below 0.8 and stayed there for 10 seconds
  - throughput_fraction recovered above 0.95 and stayed there for 60 seconds

When any of these fires, the selector rebuilds the plan.

The selector reads three inputs:

  - The ModelManifest from kiln-models
  - The current PerformanceEnvelope from the monitor
  - The Registry from kiln-hal

It produces one output:

  - An ExecutionPlan, a DAG of nodes and edges

The selection algorithm proceeds in five steps:

  Step 1. Read the model's architecture from the manifest. Dense models
          and MoE models take different paths. SSM models take a third path.

  Step 2. Read the current envelope. This constrains which plans are
          feasible. If the thermal state is SeverelyThrottling, only the
          lowest rung of the degradation ladder is available.

  Step 3. For each transformer layer or SSM block in the model, choose
          an algorithm, a precision, and a placement. The choice is driven
          by four rules:

            Rule A. If the layer is an attention layer, place it on the
                    fastest backend that supports attention at the current
                    precision and thermal state.

            Rule B. If the layer is an expert layer and the model is MoE,
                    and the expert cache hit rate target cannot be met
                    under the current plan, switch to MoEStream with
                    expert-contiguous prefetch.

            Rule C. If the layer is dense and the model is on a CPU-only
                    tier, use Diffusion if the prompt length and expected
                    output length both exceed the crossover threshold
                    from Phase 0A Research Item 1.

            Rule D. If the layer is dense and the model is on a GPU tier,
                    use Dense.

  Step 4. Assemble the DAG. The DAG is a linear chain of nodes for a
          transformer. For MoE, the expert layers become subgraphs with
          a router node feeding N expert nodes and a merge node.
          For SSM, the block is a single node.

  Step 5. Compute total estimated latency and total estimated memory.
          If either exceeds the current envelope, degrade the plan along
          the ladder and recompute. The ladder is:

            100 to 80 percent of baseline: no change
            80 to 60 percent: disable speculative decoding
            60 to 40 percent: drop KV cache precision from q8_0 to q4_0
            40 to 25 percent: switch to smallest model in catalog
            below 25 percent: pause and notify the user

The selector records every plan revision in the change log. The change log
is a Vec<PlanRevision> held by the scheduler.

## Subsystem 3 — DAG Scheduler

The scheduler owns the ExecutionPlan and the change log. It executes the
plan node by node, in topological order, on the backends assigned by the
selector. It reads the monitor before every node to check for a thermal
event that requires plan revision.

A node is executed by:

  Step 1. If the node's inputs are not resident in the required tier,
          issue a load request to the MemoryManager. The load uses the
          IoSource from kiln-io.

  Step 2. Dispatch the node's operation to the assigned backend.

  Step 3. If the node produces outputs that the next node needs in a
          different tier, mark the outputs for prefetch.

  Step 4. Record the node's actual cost in the change log.

The scheduler is single-threaded. All concurrency lives in the I/O layer
and the monitor thread. This keeps the scheduler deterministic. It is
easier to reason about the plan when the plan is executed serially.

When the monitor fires a thermal event, the scheduler completes the
current node, reads the current envelope, and asks the selector for a
new plan. The new plan replaces the old one. In-flight work completes
under the old plan. New work uses the new plan.

## Data structures

ExecutionPlan:

  struct ExecutionPlan {
      revision: u64,
      nodes: Vec<Node>,
      edges: Vec<Edge>,
      total_estimated_latency_nanos: u64,
      total_estimated_memory_bytes: u64,
      created_at: SystemTime,
  }

Node:

  struct Node {
      id: NodeId,
      op: Op,
      precision: Precision,
      placement: BackendId,
      algorithm: Algorithm,
      estimated_cost_nanos: u64,
      estimated_memory_bytes: u64,
      inputs: Vec<TensorRef>,
      outputs: Vec<TensorRef>,
  }

Edge:

  struct Edge {
      from: NodeId,
      to: NodeId,
      tensor: TensorRef,
      transfer_backend: BackendId,
      transfer_cost_nanos: u64,
  }

TensorRef:

  struct TensorRef {
      tensor: TensorId,
      precision: Precision,
      residency: ResidencyState,
      byte_size: u64,
  }

PlanRevision:

  struct PlanRevision {
      revision: u64,
      reason: RevisionReason,
      timestamp: SystemTime,
      previous_throughput_fraction: f32,
      current_throughput_fraction: f32,
  }

RevisionReason is one of: ThermalThrottle, ThermalRecovery, ManualOverride,
LoadTime, ErrorFallback.

## Interfaces

The crate exposes three public traits:

  trait PerformanceMonitor {
      fn current_envelope(&self) -> PerformanceEnvelope;
      fn thermal_history(&self, window: Duration) -> ThermalHistory;
      fn predicted_throttle(&self) -> Option<ThrottlePrediction>;
  }

  trait ModeSelector {
      fn select(&self, manifest: &ModelManifest, envelope: &PerformanceEnvelope, registry: &dyn Registry) -> ExecutionPlan;
  }

  trait Scheduler {
      fn plan(&self) -> &ExecutionPlan;
      fn revisions(&self) -> &[PlanRevision];
      fn step(&mut self) -> Result<StepOutcome, SchedulerError>;
  }

The Scheduler is stepped, not looped. Step advances the plan by one node
and returns the outcome. The caller decides when to step. This makes the
scheduler usable both in interactive mode and in the benchmark harness.

## Testing strategy

Every subsystem has a mock. The monitor is mocked by a trait object that
returns a fixed envelope. The selector is tested against a matrix of
manifest, envelope, and registry combinations. The scheduler is tested
against synthetic DAGs.

The tests do not require real hardware. The tests do not require real
models. The tests run on every platform.

## Open questions

  - How does the monitor detect SSM vs transformer at load time?
  - What is the initial baseline throughput against which throughput_fraction
    is computed? Candidates: a fixed hardware constant, a one-time
    calibration run at first load, or the average of the first N seconds
    of the current session.
  - How does the scheduler handle an error from a backend? Retry, degrade,
    or abort. The current design says degrade. The implementation may
    need to distinguish transient errors from permanent ones.
  - How are expert subgraphs scheduled when the expert cache hit rate is
    below target mid-execution? The selector uses a target hit rate, but
    the scheduler sees the actual hit rate. These two must reconcile.

## Version history

2026-10-03 — Initial draft. Three subsystems, interfaces, data structures,
testing strategy, and four open questions.
