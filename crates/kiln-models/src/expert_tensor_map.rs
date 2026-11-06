//! Explicit, validated GGUF tensor -> expert component mapping.
//!
//! GGUF's expert layouts are model-specific. Never guess an expert split from
//! a fused MoE tensor or silently treat one projection as a complete expert.
//! The caller names every tensor, and this produces checked source byte ranges
//! without copying or modifying the original GGUF.

use std::collections::HashSet;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::gguf::{GgufFile, GgufTensorInfo, GgufType};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NamedExpertTensor {
    pub layer: u32,
    pub expert_id: u32,
    pub tensor_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpertTensorComponent {
    pub layer: u32,
    pub expert_id: u32,
    pub tensor_name: String,
    pub source_byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GgufExpertSourceMap {
    pub format_version: u32,
    pub source_size_bytes: u64,
    /// Ordered by (layer, expert_id, tensor_name), not source offset.
    pub components: Vec<ExpertTensorComponent>,
}

#[derive(Debug)]
pub enum ExpertMapError {
    Invalid(String),
    Gguf(crate::gguf::GgufError),
}
impl std::fmt::Display for ExpertMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid expert tensor mapping: {s}"),
            Self::Gguf(e) => write!(f, "GGUF: {e}"),
        }
    }
}
impl std::error::Error for ExpertMapError {}
impl From<crate::gguf::GgufError> for ExpertMapError {
    fn from(e: crate::gguf::GgufError) -> Self { Self::Gguf(e) }
}
fn invalid(s: impl Into<String>) -> ExpertMapError {
    ExpertMapError::Invalid(s.into())
}

/// Unlike the legacy GGUF byte_size helper, all arithmetic here checks for
/// overflow before a mapping is accepted. Unknown quantizers are rejected.
fn checked_tensor_bytes(info: &GgufTensorInfo) -> Option<u64> {
    if info.shape.is_empty() || info.shape.contains(&0) { return None; }
    let elements = info.shape.iter().try_fold(1u64, |n, d| n.checked_mul(*d))?;
    match info.dtype {
        GgufType::F32 => elements.checked_mul(4),
        GgufType::F16 => elements.checked_mul(2),
        GgufType::Tq1_0 => elements.checked_add(4).map(|n| n / 5),
        GgufType::Unknown(_) => None,
        ref dtype => {
            let block = dtype.block_size()?;
            let bytes = dtype.block_bytes()?;
            let count = elements.checked_add(block - 1)? / block;
            count.checked_mul(bytes)
        }
    }
}

/// Validate an explicitly supplied list of exact tensor names and return
/// source ranges. A source GGUF may contain arbitrary other tensors; only
/// caller-selected names appear in the map.
pub fn resolve_expert_tensor_components(
    tensors: &[GgufTensorInfo],
    data_offset: u64,
    source_size: u64,
    names: &[NamedExpertTensor],
) -> Result<GgufExpertSourceMap, ExpertMapError> {
    if names.is_empty() || names.len() > 1_000_000 {
        return Err(invalid("expert tensor list must contain 1..=1,000,000 names"));
    }
    let by_name: std::collections::HashMap<_, _> = tensors.iter()
        .map(|tensor| (tensor.name.as_str(), tensor)).collect();
    if by_name.len() != tensors.len() {
        return Err(invalid("source GGUF contains duplicate tensor names"));
    }
    let mut seen_names = HashSet::new();
    let mut components = Vec::with_capacity(names.len());
    for selection in names {
        if !seen_names.insert(selection.tensor_name.as_str()) {
            return Err(invalid(format!("duplicate or reassigned tensor name {}",selection.tensor_name)));
        }
        let tensor = by_name.get(selection.tensor_name.as_str())
            .ok_or_else(|| invalid(format!("source GGUF lacks tensor {}",selection.tensor_name)))?;
        let len = checked_tensor_bytes(tensor).filter(|size| *size > 0)
            .ok_or_else(|| invalid(format!("unsupported/overflowing tensor {}",selection.tensor_name)))?;
        let offset = data_offset.checked_add(tensor.offset)
            .ok_or_else(|| invalid("tensor file offset overflow"))?;
        let end = offset.checked_add(len)
            .ok_or_else(|| invalid("tensor file range overflow"))?;
        if end > source_size {
            return Err(invalid(format!("tensor {} exceeds GGUF bounds",selection.tensor_name)));
        }
        components.push(ExpertTensorComponent {
            layer:selection.layer, expert_id:selection.expert_id,
            tensor_name:selection.tensor_name.clone(),
            source_byte_offset:offset, byte_length:len,
        });
    }
    let mut by_offset: Vec<_> = components.iter().collect();
    by_offset.sort_unstable_by_key(|c| c.source_byte_offset);
    for pair in by_offset.windows(2) {
        if pair[0].source_byte_offset + pair[0].byte_length > pair[1].source_byte_offset {
            return Err(invalid("selected tensor source ranges overlap"));
        }
    }
    components.sort_unstable_by(|a,b|
        (a.layer,a.expert_id,a.tensor_name.as_str())
            .cmp(&(b.layer,b.expert_id,b.tensor_name.as_str())));
    Ok(GgufExpertSourceMap {
        format_version:1, source_size_bytes:source_size, components,
    })
}

/// This mmap-based GGUF index access does not allocate a second full model
/// buffer, but the OS can still page model bytes into RAM during parsing.
pub fn resolve_expert_tensors_in_gguf(
    path: &Path, names: &[NamedExpertTensor],
) -> Result<GgufExpertSourceMap, ExpertMapError> {
    let model=GgufFile::open(path)?;
    resolve_expert_tensor_components(
        &model.tensors,model.data_offset,model.file_size(),names
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(name:&str,offset:u64,shape:&[u64],dtype:GgufType)->GgufTensorInfo {
        GgufTensorInfo{
            name:name.to_owned(),offset,shape:shape.to_vec(),dtype
        }
    }
    #[test]
    fn resolves_exact_tq1_and_q4k_source_ranges_without_repacking() {
        let tensors=vec![
            tensor("blk.0.expert.0",0,&[5,10],GgufType::Tq1_0),
            tensor("blk.0.expert.1",128,&[256],GgufType::Q4_K),
        ];
        let selections=[
            NamedExpertTensor{layer:0,expert_id:1,tensor_name:"blk.0.expert.1".into()},
            NamedExpertTensor{layer:0,expert_id:0,tensor_name:"blk.0.expert.0".into()},
        ];
        let result=resolve_expert_tensor_components(&tensors,1024,4096,&selections).unwrap();
        assert_eq!(result.components.len(),2);
        assert_eq!(result.components[0].byte_length,10);
        assert_eq!(result.components[1].byte_length,144);
        assert_eq!(result.components[0].source_byte_offset,1024);
        assert_eq!(result.components[1].source_byte_offset,1152);
    }
    #[test]
    fn rejects_overlaps_missing_unknown_and_overflow() {
        let tensors=vec![
            tensor("a",0,&[u64::MAX,4],GgufType::F16),
            tensor("b",1,&[32],GgufType::F32),
            tensor("c",10,&[32],GgufType::F32),
        ];
        let selector=|name:&str| NamedExpertTensor{
            layer:0,expert_id:0,tensor_name:name.into()
        };
        assert!(resolve_expert_tensor_components(&tensors,0,1000,&[selector("a")]).is_err());
        assert!(resolve_expert_tensor_components(&tensors,0,1000,&[selector("missing")]).is_err());
        assert!(resolve_expert_tensor_components(&tensors,0,1000,
            &[selector("b"),selector("c")]).is_err());
        assert!(resolve_expert_tensor_components(&tensors,0,1000,
            &[selector("b"),selector("b")]).is_err());
    }
    #[test]
    fn rejects_unsupported_quant_and_out_of_bounds() {
        let tensor_a=tensor("a",0,&[256],GgufType::Unknown(999));
        let select=NamedExpertTensor{layer:0,expert_id:0,tensor_name:"a".into()};
        assert!(resolve_expert_tensor_components(&[tensor_a],0,4096,&[select.clone()]).is_err());
        let tensor_b=tensor("a",5,&[64],GgufType::F32);
        assert!(resolve_expert_tensor_components(&[tensor_b],0,128,&[select]).is_err());
    }
}
