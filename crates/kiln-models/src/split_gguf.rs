//! Read-only, per-shard source mapping for split GGUF models.
//!
//! This inventories actual tensor source files without concatenating weights
//! or pretending the single-blob native MoE index supports multiple files.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use crate::gguf::{GgufFile, GgufTensorInfo, GgufType};
use crate::library::{normalize_gguf_load_path, split_gguf_parts};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SplitExpertSource {
    pub tensor_name: String,
    pub shard_number: u32,
    pub source_path: PathBuf,
    pub source_byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SplitShardSummary {
    pub shard_number: u32,
    pub path: PathBuf,
    pub file_size_bytes: u64,
    pub tensor_count: usize,
    pub expert_tensor_count: usize,
    pub unsupported_expert_tensor_count: usize,
}

/// One fully validated, original-file expert slab of a known fused
/// projection. Selection uses the declared third GGUF dimension, never an
/// inferred weight axis or a guessed byte division.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SplitFusedExpertSlab {
    pub tensor_name: String,
    pub layer: u32,
    pub expert_id: u32,
    pub shard_number: u32,
    pub source_path: PathBuf,
    pub source_byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitGgufInventory {
    pub entrypoint: PathBuf,
    pub shard_count: u32,
    pub total_file_bytes: u64,
    pub total_tensor_count: usize,
    pub architecture: Option<String>,
    pub shards: Vec<SplitShardSummary>,
    /// Exact validated source bytes for recognized expert tensors. This is
    /// evidence for future multi-source streaming; not a runnable index.
    pub expert_sources: Vec<SplitExpertSource>,
    /// Complete uniform 3-projection fused expert tensor slices across
    /// every shard; empty means no verified full-model fused layout.
    pub fused_expert_slabs: Vec<SplitFusedExpertSlab>,
    pub unsupported_expert_tensors: Vec<String>,
    pub native_multi_source_streaming_ready: bool,
}

fn is_expert_tensor(name: &str) -> bool {
    name.contains("_exps") || name.contains(".expert.")
}

/// Validate source offsets without relying on the legacy unchecked GGUF
/// num_elements/byte_size helpers. Unknown formats remain unsupported.
fn checked_tensor_byte_length(tensor: &GgufTensorInfo) -> Option<u64> {
    let width = *tensor.shape.first()?;
    if width == 0 || tensor.shape.iter().any(|n| *n == 0) { return None; }
    let rows = tensor.shape[1..].iter().try_fold(1u64, |acc, n| acc.checked_mul(*n))?;
    let row_bytes = match tensor.dtype {
        GgufType::F16 => width.checked_mul(2)?,
        GgufType::F32 => width.checked_mul(4)?,
        GgufType::Tq1_0 => width.checked_add(4)? / 5,
        GgufType::Unknown(_) => return None,
        ref dtype => {
            let block = dtype.block_size()?;
            if width % block != 0 { return None; }
            (width / block).checked_mul(dtype.block_bytes()?)?
        }
    };
    row_bytes.checked_mul(rows)
}

fn expert_source(
    tensor: &GgufTensorInfo, shard_number: u32, path: &Path,
    data_offset: u64, file_size: u64
) -> Option<SplitExpertSource> {
    let size = checked_tensor_byte_length(tensor).filter(|n| *n > 0)?;
    let absolute = data_offset.checked_add(tensor.offset)?;
    let end = absolute.checked_add(size)?;
    if end > file_size { return None; }
    Some(SplitExpertSource {
        tensor_name: tensor.name.clone(),
        shard_number,
        source_path: path.to_path_buf(),
        source_byte_offset: absolute,
        byte_length: size,
    })
}

fn fused_projection(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, projection) = rest.split_once('.')?;
    if !matches!(projection, "ffn_up_exps.weight" | "ffn_down_exps.weight" |
        "ffn_gate_exps.weight") { return None; }
    // Reject leading zeros and malformed layer names instead of allowing
    // multiple spellings to bypass per-layer uniqueness checks.
    let layer_number: u32 = layer.parse().ok()?;
    if layer_number.to_string() != layer { return None; }
    Some((layer_number, projection))
}

/// Strictly recognize complete, uniform third-axis experts across all source
/// shards, using only declared metadata from the first GGUF. Each tensor's
/// slabs remain in the original file at checked absolute offsets.
fn fused_slabs(
    architecture: Option<&str>,
    metadata: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    tensors: &[(u32, PathBuf, u64, u64, GgufTensorInfo)]
) -> Result<Vec<SplitFusedExpertSlab>, String> {
    let Some(arch) = architecture else { return Ok(Vec::new()); };
    let expert_count = metadata.get(&format!("{arch}.expert_count"))
        .and_then(|v| v.as_u32()).filter(|n| (2..=65536).contains(n));
    let block_count = metadata.get(&format!("{arch}.block_count"))
        .and_then(|v| v.as_u32()).filter(|n| (1..=10000).contains(n));
    let (Some(expert_count),Some(block_count)) = (expert_count,block_count) else {
        return Ok(Vec::new());
    };
    let mut selections = Vec::new();
    let mut seen = HashSet::new();
    for (shard,path,data_offset,file_size,tensor) in tensors {
        let Some((layer,projection)) = fused_projection(&tensor.name) else { continue; };
        if layer >= block_count {
            return Err(format!("fused tensor {} layer exceeds declared block count",tensor.name));
        }
        if !seen.insert((layer,projection)) {
            return Err(format!("duplicate fused projection {} in the split GGUF",tensor.name));
        }
        if tensor.shape.len()!=3 || tensor.shape[2]!=u64::from(expert_count) {
            return Err(format!("fused tensor {} has inconsistent expert axis",tensor.name));
        }
        let mut slice = tensor.clone();
        slice.shape.truncate(2);
        let slab_bytes = checked_tensor_byte_length(&slice)
            .filter(|n| *n>0).ok_or_else(||format!(
                "unsupported fused expert tensor layout {}",tensor.name))?;
        let full = slab_bytes.checked_mul(u64::from(expert_count))
            .ok_or("fused expert tensor byte-length overflow")?;
        if checked_tensor_byte_length(tensor) != Some(full) {
            return Err(format!("fused tensor {} has noncontiguous expert slabs",tensor.name));
        }
        let first = data_offset.checked_add(tensor.offset)
            .ok_or("fused tensor absolute offset overflow")?;
        let end = first.checked_add(full).ok_or("fused tensor end overflow")?;
        if end > *file_size { return Err(format!(
            "fused tensor {} exceeds its source shard",tensor.name)); }
        for expert_id in 0..expert_count {
            let offset = first.checked_add(slab_bytes*u64::from(expert_id))
                .ok_or("fused expert slab offset overflow")?;
            selections.push(SplitFusedExpertSlab {
                tensor_name:tensor.name.clone(),layer,expert_id,
                shard_number:*shard,source_path:path.clone(),
                source_byte_offset:offset,byte_length:slab_bytes,
            });
        }
    }
    let wanted = usize::try_from(block_count).ok().and_then(|n| n.checked_mul(3))
        .ok_or("fused expert block count overflow")?;
    if seen.len() != wanted {
        // A partial model or a mixed MoE architecture is not certified as
        // complete; do not export runnable native stream selections.
        return Ok(Vec::new());
    }
    selections.sort_unstable_by(|a,b|
        (a.layer,a.expert_id,a.tensor_name.as_str()).cmp(&
         (b.layer,b.expert_id,b.tensor_name.as_str())));
    Ok(selections)
}

/// Inspect all numbered parts in-place; never download, rewrite or combine
/// the original shards. Each expert source is tied to one exact shard.
pub fn inspect_split_gguf(path: &Path) -> Result<Option<SplitGgufInventory>, String> {
    let Some((_, _, count)) = split_gguf_parts(path) else { return Ok(None); };
    let first = normalize_gguf_load_path(path)?;
    let (prefix, _, _) = split_gguf_parts(&first)
        .ok_or("invalid split GGUF entrypoint")?;
    let base = prefix.file_name().and_then(|n| n.to_str())
        .ok_or("split GGUF prefix is not UTF-8")?;
    let parent = prefix.parent().ok_or("missing split directory")?;
    let mut shards = Vec::with_capacity(count as usize);
    let mut sources = Vec::new();
    let mut unsupported = Vec::new();
    let mut names = HashSet::new();
    let mut total_bytes = 0u64;
    let mut total_tensors = 0usize;
    let mut architecture: Option<String> = None;
    let mut first_metadata = None;
    let mut all_tensors = Vec::new();

    for number in 1..=count {
        let shard = parent.join(format!("{base}-{number:05}-of-{count:05}.gguf"));
        let model = GgufFile::open(&shard)
            .map_err(|e| format!("GGUF shard {number}/{count} ({}): {e}", shard.display()))?;
        let expected_split_number = number - 1;
        if let Some(encoded) = model.metadata.get("split.no") {
            if encoded.as_u32() != Some(expected_split_number) {
                return Err(format!("shard {} has incorrect split.no (expected {})",
                    shard.display(),expected_split_number));
            }
        }
        if let Some(encoded) = model.metadata.get("split.count") {
            if encoded.as_u32() != Some(count) {
                return Err(format!("shard {} declares a different split.count",shard.display()));
            }
        }
        let shard_arch = model.metadata_str("general.architecture").map(str::to_owned);
        // Upstream llama-gguf-split intentionally writes the original
        // architecture metadata ONLY to shard 1. Later shards normally
        // contain split.no/split.count and tensor tables without it.
        if number == 1 {
            architecture = shard_arch;
        } else if let Some(ref explicit) = shard_arch {
            if architecture.as_ref() != Some(explicit) {
                return Err(format!("split GGUF shards disagree on architecture: {}",
                    shard.display()));
            }
        }
        if number == 1 { first_metadata = Some(model.metadata.clone()); }
        total_bytes = total_bytes.checked_add(model.file_size())
            .ok_or("split GGUF total file size overflow")?;
        total_tensors = total_tensors.checked_add(model.tensor_count())
            .ok_or("split GGUF total tensor count overflow")?;
        if total_tensors > 1_000_000 {
            return Err("split GGUF exceeds the one-million-tensor validation limit".into());
        }
        let mut expert_count = 0;
        let mut unsupported_count = 0;
        for tensor in &model.tensors {
            if is_expert_tensor(&tensor.name) {
                all_tensors.push((number,shard.clone(),model.data_offset,
                    model.file_size(),tensor.clone()));
            }
            if !names.insert(tensor.name.as_str().to_owned()) {
                return Err(format!("duplicate tensor {} across GGUF shards",tensor.name));
            }
            if !is_expert_tensor(&tensor.name) { continue; }
            expert_count += 1;
            match expert_source(tensor,number,&shard,model.data_offset,model.file_size()) {
                Some(source) => sources.push(source),
                None => {
                    unsupported_count += 1;
                    if unsupported.len() < 128 {
                        unsupported.push(format!("{}: {}",shard.display(),tensor.name));
                    }
                }
            }
        }
        shards.push(SplitShardSummary {
            shard_number:number, path:shard, file_size_bytes:model.file_size(),
            tensor_count:model.tensor_count(), expert_tensor_count:expert_count,
            unsupported_expert_tensor_count:unsupported_count,
        });
    }
    let fused_expert_slabs = fused_slabs(architecture.as_deref(),
        first_metadata.as_ref().ok_or("first shard metadata missing")?,
        &all_tensors)?;
    Ok(Some(SplitGgufInventory {
        entrypoint:first, shard_count:count,total_file_bytes:total_bytes,
        total_tensor_count:total_tensors, architecture,shards,
        expert_sources:sources, fused_expert_slabs,
        unsupported_expert_tensors:unsupported,
        // The native GGML expert reader still supports one original GGUF
        // source path. A validated per-shard map is NOT a working multi-file
        // compute path, so this must remain false until native integration.
        native_multi_source_streaming_ready:false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(dtype:GgufType, shape:Vec<u64>, offset:u64)->GgufTensorInfo{
        GgufTensorInfo{name:"blk.0.ffn_up_exps.weight".into(),dtype,shape,offset}
    }


    fn push_string(data: &mut Vec<u8>, value: &str) {
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }

    fn tiny_split_shard(path: &Path, part: u32, tensor_name: &str, dtype: u32) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&3u64.to_le_bytes()); // three metadata keys
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string
        push_string(&mut bytes, "tiny-moe");
        push_string(&mut bytes, "split.no");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // uint32
        bytes.extend_from_slice(&(part-1).to_le_bytes());
        push_string(&mut bytes, "split.count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        push_string(&mut bytes, tensor_name);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // dimensions
        for dim in [32u64, 2, 2] { bytes.extend_from_slice(&dim.to_le_bytes()); }
        bytes.extend_from_slice(&dtype.to_le_bytes()); // F16 or unknown
        bytes.extend_from_slice(&0u64.to_le_bytes());
        while bytes.len() % 32 != 0 { bytes.push(0); }
        bytes.extend_from_slice(&vec![0u8; 256]);
        std::fs::write(path,bytes).unwrap();
    }

    #[test]
    fn inventories_two_real_minimal_gguf_shards_without_merging_or_modifying() {
        let unique=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("kiln-split-source-{}-{unique}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let first=dir.join("tiny-00001-of-00002.gguf");
        let second=dir.join("tiny-00002-of-00002.gguf");
        tiny_split_shard(&first,1,"blk.0.ffn_up_exps.weight",1);
        tiny_split_shard(&second,2,"blk.0.ffn_down_exps.weight",1);
        let initial=std::fs::read(&first).unwrap();
        let found=inspect_split_gguf(&second).unwrap().unwrap();
        assert_eq!(found.entrypoint,first);
        assert_eq!(found.shard_count,2);
        assert_eq!(found.total_tensor_count,2);
        assert_eq!(found.expert_sources.len(),2);
        assert_eq!(found.expert_sources[0].shard_number,1);
        assert_eq!(found.expert_sources[1].shard_number,2);
        assert_eq!(found.expert_sources[1].byte_length,256);
        assert!(!found.native_multi_source_streaming_ready);
        assert_eq!(std::fs::read(&first).unwrap(),initial);
        std::fs::remove_file(&second).unwrap();
        assert!(inspect_split_gguf(&first).unwrap_err().contains("00002-of-00002"));
        tiny_split_shard(&second,2,"blk.0.ffn_up_exps.weight",1);
        assert!(inspect_split_gguf(&first).unwrap_err().contains("duplicate tensor"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_expert_quantizer_is_counted_but_not_accepted_for_source_mapping() {
        let unique=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("kiln-split-unknown-{}-{unique}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let first=dir.join("tiny-00001-of-00002.gguf");
        let second=dir.join("tiny-00002-of-00002.gguf");
        tiny_split_shard(&first,1,"blk.0.ffn_up_exps.weight",1);
        tiny_split_shard(&second,2,"blk.0.ffn_down_exps.weight",12345);
        let found=inspect_split_gguf(&first).unwrap().unwrap();
        assert_eq!(found.expert_sources.len(),1);
        assert_eq!(found.unsupported_expert_tensors.len(),1);
        assert_eq!(found.shards[1].unsupported_expert_tensor_count,1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn complete_fused_expert_slabs_span_multiple_shards_without_ambiguity() {
        use crate::gguf::GgufValue;
        use std::collections::HashMap;
        let mut meta=HashMap::new();
        meta.insert("tiny.expert_count".into(),GgufValue::Uint32(2));
        meta.insert("tiny.block_count".into(),GgufValue::Uint32(1));
        let tensors=["ffn_up_exps.weight","ffn_gate_exps.weight","ffn_down_exps.weight"]
            .iter().enumerate().map(|(i,proj)|{
                let t=GgufTensorInfo{name:format!("blk.0.{proj}"),
                    dtype:GgufType::F16,shape:vec![4,3,2],offset:0};
                (i as u32 % 2 + 1,PathBuf::from(format!("/shard-{i}.gguf")),32,1000,t)
            }).collect::<Vec<_>>();
        let full=fused_slabs(Some("tiny"),&meta,&tensors).unwrap();
        assert_eq!(full.len(),6);
        assert_eq!(full.iter().filter(|x| x.expert_id==0).count(),3);
        assert!(full.iter().any(|x| x.shard_number==2));
        for item in full.iter().filter(|item| item.tensor_name=="blk.0.ffn_up_exps.weight") {
            assert_eq!(item.byte_length,24);
            assert_eq!(item.source_byte_offset,32+u64::from(item.expert_id)*24);
        }
        assert!(fused_slabs(Some("tiny"),&meta,&tensors[..2]).unwrap().is_empty());
        let mut invalid=tensors.clone();
        invalid[1].4.shape[2]=3;
        assert!(fused_slabs(Some("tiny"),&meta,&invalid).is_err());
    }

    #[test]
    fn checks_row_boundaries_and_overflow_before_expert_mapping() {
        let quant=tensor(GgufType::Q4_K,vec![256,4,8],0);
        assert_eq!(checked_tensor_byte_length(&quant),Some(144*4*8));
        let partial=tensor(GgufType::Q4_K,vec![255,4,8],0);
        assert_eq!(checked_tensor_byte_length(&partial),None);
        assert_eq!(checked_tensor_byte_length(&tensor(GgufType::Unknown(42),vec![256,8],0)),None);
        assert_eq!(checked_tensor_byte_length(&tensor(GgufType::F32,vec![u64::MAX,4],0)),None);
        assert_eq!(checked_tensor_byte_length(&tensor(GgufType::Tq1_0,vec![9,2],0)),Some(4));
    }

    #[test]
    fn expert_source_is_file_scoped_and_rejects_out_of_bounds() {
        let entry=tensor(GgufType::F16,vec![10,3],16);
        let file=Path::new("/models/model-00002-of-00002.gguf");
        let map=expert_source(&entry,2,file,128,2048).unwrap();
        assert_eq!(map.source_path,file);
        assert_eq!(map.shard_number,2);
        assert_eq!(map.source_byte_offset,144);
        assert_eq!(map.byte_length,60);
        assert!(expert_source(&entry,2,file,128,160).is_none());
        assert!(expert_source(&entry,2,file,u64::MAX,2048).is_none());
    }
}
