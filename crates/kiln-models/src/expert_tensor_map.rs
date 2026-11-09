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
    // GGUF dimensions are recorded with the contiguous row dimension first.
    // Quantization blocks are encoded independently in each row; flattening
    // dimensions before ceil-dividing undercounts weights with partial rows.
    let row_width = *info.shape.first()?;
    let row_count = info.shape[1..].iter().try_fold(1u64, |n,d| n.checked_mul(*d))?;
    let row_bytes = match info.dtype {
        GgufType::F32 => row_width.checked_mul(4)?,
        GgufType::F16 => row_width.checked_mul(2)?,
        GgufType::Tq1_0 => row_width.checked_add(4)? / 5,
        GgufType::Unknown(_) => return None,
        ref dtype => {
            let block = dtype.block_size()?;
            let bytes = dtype.block_bytes()?;
            // Avoid accepting a partial block that the pinned native format
            // cannot decode as a valid row. GGUF quantized row width must be
            // an integral number of blocks, except for KILN TQ1.0 packing.
            if row_width % block != 0 { return None; }
            row_width.checked_div(block)?.checked_mul(bytes)?
        }
    };
    row_bytes.checked_mul(row_count)
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

/// A GGUF fused-expert tensor explicitly declared to use contiguous
/// expert slices on the THIRD GGUF dimension: [row_width, row_count, experts].
/// Never infer an expert axis from a tensor name or metadata alone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FusedExpertTensorSelection {
    pub layer: u32,
    pub tensor_name: String,
    pub expert_count: u32,
}

/// Resolve byte-exact contiguous per-expert slabs of declared 3-D fused
/// tensors. This is a storage transform only: no inference kernel is changed.
/// Unsupported formats and ambiguous axes are rejected before reading bytes.
pub fn resolve_fused_expert_tensor_components(
    tensors: &[GgufTensorInfo],
    data_offset: u64,
    source_size: u64,
    selections: &[FusedExpertTensorSelection],
) -> Result<GgufExpertSourceMap, ExpertMapError> {
    if selections.is_empty() || selections.len() > 100_000 {
        return Err(invalid("fused selection count must be 1..=100,000"));
    }
    let by_name: std::collections::HashMap<_, _> = tensors.iter()
        .map(|tensor| (tensor.name.as_str(), tensor)).collect();
    if by_name.len() != tensors.len() {
        return Err(invalid("source GGUF contains duplicate tensor names"));
    }
    let mut seen = HashSet::new();
    let mut components = Vec::new();
    for selection in selections {
        if !seen.insert(selection.tensor_name.as_str()) {
            return Err(invalid(format!("duplicate fused tensor {}", selection.tensor_name)));
        }
        if selection.expert_count == 0 ||
            components.len().checked_add(selection.expert_count as usize)
                .is_none_or(|count| count > 1_000_000) {
            return Err(invalid("fused selection count or total expert components out of bounds"));
        }
        let tensor = by_name.get(selection.tensor_name.as_str())
            .ok_or_else(|| invalid(format!("source GGUF lacks fused tensor {}", selection.tensor_name)))?;
        if tensor.shape.len() != 3 || tensor.shape[2] != u64::from(selection.expert_count) {
            return Err(invalid(format!(
                "fused tensor {} must have exact [row_width, rows, {}] dimensions",
                selection.tensor_name, selection.expert_count
            )));
        }
        let slice_info = GgufTensorInfo {
            name: tensor.name.clone(),
            shape: tensor.shape[..2].to_vec(),
            dtype: tensor.dtype,
            offset: tensor.offset,
        };
        let slab_bytes = checked_tensor_bytes(&slice_info)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| invalid(format!("unsupported fused tensor layout {}", selection.tensor_name)))?;
        let full_bytes = slab_bytes.checked_mul(u64::from(selection.expert_count))
            .ok_or_else(|| invalid("fused tensor size overflow"))?;
        if checked_tensor_bytes(tensor) != Some(full_bytes) {
            return Err(invalid("fused expert slabs do not match total encoded tensor bytes"));
        }
        let tensor_start = data_offset.checked_add(tensor.offset)
            .ok_or_else(|| invalid("fused tensor offset overflow"))?;
        let tensor_end = tensor_start.checked_add(full_bytes)
            .ok_or_else(|| invalid("fused tensor range overflow"))?;
        if tensor_end > source_size {
            return Err(invalid(format!("fused tensor {} exceeds source bounds", tensor.name)));
        }
        for expert_id in 0..selection.expert_count {
            let offset = tensor_start.checked_add(slab_bytes * u64::from(expert_id))
                .ok_or_else(|| invalid("expert offset overflow"))?;
            components.push(ExpertTensorComponent {
                layer: selection.layer,
                expert_id,
                // Make each slab's component identity explicit; it is a slice of
                // the original tensor rather than a standalone GGUF tensor.
                tensor_name: format!("{}#expert:{}", tensor.name, expert_id),
                source_byte_offset: offset,
                byte_length: slab_bytes,
            });
        }
    }
    let mut by_offset: Vec<_> = components.iter().collect();
    by_offset.sort_unstable_by_key(|item| item.source_byte_offset);
    for pair in by_offset.windows(2) {
        if pair[0].source_byte_offset + pair[0].byte_length > pair[1].source_byte_offset {
            return Err(invalid("fused expert source ranges overlap"));
        }
    }
    components.sort_unstable_by(|a,b|
        (a.layer,a.expert_id,a.tensor_name.as_str())
            .cmp(&(b.layer,b.expert_id,b.tensor_name.as_str())));
    for pair in components.windows(2) {
        if (pair[0].layer,pair[0].expert_id,pair[0].tensor_name.as_str()) ==
            (pair[1].layer,pair[1].expert_id,pair[1].tensor_name.as_str()) {
            return Err(invalid("duplicate fused expert component key"));
        }
    }
    Ok(GgufExpertSourceMap {
        format_version: 1, source_size_bytes: source_size, components,
    })
}

/// Read-only GGUF-to-fused-slab map. The model remains intact on disk.
pub fn resolve_fused_expert_tensors_in_gguf(
    path: &Path,
    selections: &[FusedExpertTensorSelection],
) -> Result<GgufExpertSourceMap, ExpertMapError> {
    let model = GgufFile::open(path)?;
    resolve_fused_expert_tensor_components(
        &model.tensors, model.data_offset, model.file_size(), selections
    )
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
    fn fused_q4k_experts_have_exact_nonoverlapping_quantized_slabs() {
        let tensors = vec![
            tensor("blk.0.ffn_up_exps",0,&[256,3,4],GgufType::Q4_K),
            tensor("blk.0.ffn_down_exps",4096,&[256,2,4],GgufType::Q4_K),
        ];
        let choices = [
            FusedExpertTensorSelection {layer:0,tensor_name:"blk.0.ffn_up_exps".into(),expert_count:4},
            FusedExpertTensorSelection {layer:0,tensor_name:"blk.0.ffn_down_exps".into(),expert_count:4},
        ];
        let result=resolve_fused_expert_tensor_components(&tensors,1024,8192,&choices).unwrap();
        assert_eq!(result.components.len(),8);
        let e1:Vec<_>=result.components.iter().filter(|x| x.expert_id==1).collect();
        assert_eq!(e1.len(),2);
        assert_eq!(e1[0].byte_length,288);
        assert_eq!(e1[0].source_byte_offset,1024+4096+288);
        assert_eq!(e1[1].byte_length,432);
        assert_eq!(e1[1].source_byte_offset,1024+432);
    }

    #[test]
    fn fused_mapper_rejects_ambiguous_axis_partial_quant_rows_and_duplicates() {
        let t = tensor("fused",0,&[256,2,4],GgufType::Q4_K);
        let select=FusedExpertTensorSelection {
            layer:0,tensor_name:"fused".into(),expert_count:4,
        };
        assert!(resolve_fused_expert_tensor_components(&[t.clone()],0,4096,&[select.clone()]).is_ok());
        assert!(resolve_fused_expert_tensor_components(&[t.clone()],0,4096,&[select.clone(),select.clone()]).is_err());
        assert!(resolve_fused_expert_tensor_components(&[t.clone()],0,4096,&[FusedExpertTensorSelection {
            expert_count:3, ..select.clone()
        }]).is_err());
        assert!(resolve_fused_expert_tensor_components(&[tensor("fused",0,&[257,2,4],GgufType::Q4_K)],0,4096,&[select.clone()]).is_err());
        assert!(resolve_fused_expert_tensor_components(&[tensor("fused",0,&[256,8],GgufType::Q4_K)],0,4096,&[select.clone()]).is_err());
        assert!(resolve_fused_expert_tensor_components(&[t],0,1,&[select]).is_err());
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
    fn tq1_partial_rows_count_each_row_separately() {
        let src=[tensor("packed",0,&[6,3],GgufType::Tq1_0)];
        let names=[NamedExpertTensor {
            layer:0,expert_id:0,tensor_name:"packed".into(),
        }];
        let map=resolve_expert_tensor_components(&src,0,128,&names).unwrap();
        // Three 6-trit rows each require ceil(6/5)=2 bytes; not
        // ceil(18/5)=4 total bytes.
        assert_eq!(map.components[0].byte_length,6);
    }

    #[test]
    fn block_quantized_rows_must_be_complete() {
        let src=[tensor("quant",0,&[257,2],GgufType::Q4_K)];
        let names=[NamedExpertTensor {
            layer:0,expert_id:0,tensor_name:"quant".into(),
        }];
        assert!(resolve_expert_tensor_components(&src,0,8192,&names).is_err());
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
