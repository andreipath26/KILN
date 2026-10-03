//! The Execution Plan DAG and its node, edge, and tensor types.

use std::time::SystemTime;

use kiln_hal::{Algorithm, BackendId, Op, Precision, ResidencyState};
use kiln_mem::TensorId;
use serde::{Deserialize, Serialize};

/// Unique identifier for a node in the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// A reference to a tensor at a specific precision and residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRef {
    pub tensor: TensorId,
    pub precision: Precision,
    pub residency: ResidencyState,
    pub byte_size: u64,
}

/// A single computation in the plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub precision: Precision,
    pub placement: BackendId,
    pub algorithm: Algorithm,
    pub estimated_cost_nanos: u64,
    pub estimated_memory_bytes: u64,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
}

/// A data transfer between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub tensor: TensorRef,
    pub transfer_backend: BackendId,
    pub transfer_cost_nanos: u64,
}

/// The full plan. A DAG of nodes and edges plus totals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub revision: u64,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub total_estimated_latency_nanos: u64,
    pub total_estimated_memory_bytes: u64,
    #[serde(with = "serde_system_time")]
    pub created_at: SystemTime,
}

impl ExecutionPlan {
    /// Find a node by id.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// All nodes that directly feed into the given node.
    pub fn parents_of(&self, id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.to == id)
            .map(|e| e.from)
            .collect()
    }
}

/// Serialization shim for SystemTime because serde does not implement it.
mod serde_system_time {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let d = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        d.as_secs().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}
