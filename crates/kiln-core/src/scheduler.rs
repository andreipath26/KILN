//! The DAG Scheduler.
//!
//! Executes the ExecutionPlan node by node, in topological order. Reads the
//! monitor before every node to check for a thermal event. When a thermal
//! event fires, it revokes the current plan and asks the selector for a new
//! one. Every revision is recorded in an append-only change log.
//!
//! See docs/core-design.md, Subsystem 3.

use std::time::SystemTime;

use kiln_hal::Registry;

use crate::error::SchedulerError;
use crate::monitor::{PerformanceEnvelope, PerformanceMonitor};
use crate::plan::{ExecutionPlan, NodeId};
use crate::revision::{PlanRevision, RevisionReason};
use crate::selector::ModeSelector;

/// The outcome of one step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    /// A node was executed successfully.
    Executed { node: NodeId, duration_nanos: u64 },
    /// The plan was revised before this step. The caller should re-read the
    /// plan and continue.
    Revised { revision: u64 },
    /// The plan is complete.
    Finished,
}

/// The Scheduler trait.
pub trait Scheduler: Send + Sync {
    /// The current plan.
    fn plan(&self) -> &ExecutionPlan;

    /// The append-only change log.
    fn revisions(&self) -> &[PlanRevision];

    /// Advance the plan by one node. Returns the outcome.
    fn step(&mut self) -> Result<StepOutcome, SchedulerError>;
}

/// The default scheduler.
pub struct DefaultScheduler {
    plan: ExecutionPlan,
    cursor: usize,
    revisions: Vec<PlanRevision>,
    previous_throughput: f32,
}

impl DefaultScheduler {
    /// Build a scheduler with an initial plan.
    pub fn new(plan: ExecutionPlan, initial_throughput: f32) -> Self {
        let mut revisions = Vec::new();
        revisions.push(PlanRevision {
            revision: plan.revision,
            reason: RevisionReason::LoadTime,
            timestamp: SystemTime::now(),
            previous_throughput_fraction: 1.0,
            current_throughput_fraction: initial_throughput,
        });
        Self {
            plan,
            cursor: 0,
            revisions,
            previous_throughput: initial_throughput,
        }
    }

    /// The revision count so far.
    pub fn revision_count(&self) -> usize {
        self.revisions.len()
    }

    /// Whether the plan is complete.
    pub fn is_finished(&self) -> bool {
        self.cursor >= self.plan.nodes.len()
    }

    /// Check the monitor for a thermal event that warrants plan revision.
    /// Returns Some(reason) if a revision is warranted.
    fn check_for_revision(
        &self,
        envelope: &PerformanceEnvelope,
    ) -> Option<RevisionReason> {
        // Rule 1: thermal state changed to Throttling or worse.
        if matches!(
            envelope.thermal_state,
            kiln_hal::ThermalState::Throttling | kiln_hal::ThermalState::SeverelyThrottling
        ) && self.previous_throughput >= 0.80
        {
            return Some(RevisionReason::ThermalThrottle);
        }

        // Rule 2: throughput dropped below 0.8 and stayed there.
        if envelope.throughput_fraction < 0.80 && self.previous_throughput >= 0.80 {
            return Some(RevisionReason::ThermalThrottle);
        }

        // Rule 3: throughput recovered above 0.95 after having been below.
        if envelope.throughput_fraction >= 0.95 && self.previous_throughput < 0.80 {
            return Some(RevisionReason::ThermalRecovery);
        }

        None
    }
}

impl Scheduler for DefaultScheduler {
    fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    fn revisions(&self) -> &[PlanRevision] {
        &self.revisions
    }

    fn step(&mut self) -> Result<StepOutcome, SchedulerError> {
        // Step 0: check if finished.
        if self.cursor >= self.plan.nodes.len() {
            return Ok(StepOutcome::Finished);
        }

        // Step 1: read the monitor. The caller passes it through a shared
        // reference. In this version we use a thread-local reference that
        // the caller sets. See the test harness for usage.
        //
        // In a future revision, the scheduler will accept the monitor as a
        // step parameter. For now, the scheduler works without a live
        // monitor and only revises when explicitly asked.
        //
        // Step 2: execute the current node.
        let node = &self.plan.nodes[self.cursor];
        let duration = node.estimated_cost_nanos;
        let id = node.id;
        self.cursor += 1;
        Ok(StepOutcome::Executed { node: id, duration_nanos: duration })
    }
}

/// A scheduler that accepts an explicit monitor. Used in tests and in the
/// production loop where the monitor is passed at each step.
pub struct MonitoredScheduler<'a> {
    inner: DefaultScheduler,
    monitor: &'a dyn PerformanceMonitor,
    registry: &'a dyn Registry,
    selector: &'a dyn ModeSelector,
}

impl<'a> MonitoredScheduler<'a> {
    pub fn new(
        plan: ExecutionPlan,
        initial_throughput: f32,
        monitor: &'a dyn PerformanceMonitor,
        registry: &'a dyn Registry,
        selector: &'a dyn ModeSelector,
    ) -> Self {
        Self {
            inner: DefaultScheduler::new(plan, initial_throughput),
            monitor,
            registry,
            selector,
        }
    }

    pub fn plan(&self) -> &ExecutionPlan {
        self.inner.plan()
    }

    pub fn revisions(&self) -> &[PlanRevision] {
        self.inner.revisions()
    }

    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Step with a manifest. Reads the monitor, revises if warranted, then
    /// executes one node.
    pub fn step_with_manifest(
        &mut self,
        manifest: &kiln_models::ModelManifest,
    ) -> Result<StepOutcome, SchedulerError> {
        if self.inner.is_finished() {
            return Ok(StepOutcome::Finished);
        }

        let envelope = self.monitor.current_envelope();
        if let Some(reason) = self.inner.check_for_revision(&envelope) {
            let new_revision = self.inner.plan.revision + 1;
            let outcome = self.selector.select(
                manifest,
                &envelope,
                self.registry,
                self.monitor,
            );
            let mut new_plan = outcome.plan;
            new_plan.revision = new_revision;
            self.inner.revisions.push(PlanRevision {
                revision: new_revision,
                reason,
                timestamp: SystemTime::now(),
                previous_throughput_fraction: self.inner.previous_throughput,
                current_throughput_fraction: envelope.throughput_fraction,
            });
            self.inner.previous_throughput = envelope.throughput_fraction;
            self.inner.plan = new_plan;
            self.inner.cursor = 0;
            return Ok(StepOutcome::Revised { revision: new_revision });
        }

        self.inner.previous_throughput = envelope.throughput_fraction;
        self.inner.step()
    }
}
