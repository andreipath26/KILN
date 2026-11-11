//! Read-only source-accurate expert tensor I/O across split GGUF shards.
//! Native graph integration is separate: this does not activate multi-file
//! computation or turn a diagnostic map into a single-source native index.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use kiln_models::split_gguf::{inspect_split_gguf, SplitGgufInventory, SplitExpertSource};
use crate::FileRangeReader;

pub struct SplitGgufExpertReader {
    inventory: SplitGgufInventory,
    sources: Vec<Mutex<FileRangeReader>>,
    by_tensor_name: HashMap<String, usize>,
}

impl SplitGgufExpertReader {
    /// Open and pin each original shard as a separate descriptor.
    /// No model weights are copied, joined, or changed.
    pub fn open(path: &Path) -> Result<Option<Self>, String> {
        match inspect_split_gguf(path)? {
            Some(inventory) => Self::from_inventory(inventory).map(Some),
            None => Ok(None),
        }
    }

    fn from_inventory(inventory: SplitGgufInventory) -> Result<Self, String> {
        let mut sources = Vec::with_capacity(inventory.shards.len());
        if inventory.shards.len() != inventory.shard_count as usize {
            return Err("split inventory shard count is inconsistent".into());
        }
        for (index, shard) in inventory.shards.iter().enumerate() {
            if shard.shard_number != index as u32 + 1 {
                return Err("split source shard numbers must be contiguous".into());
            }
            let reader = FileRangeReader::open(&shard.path)
                .map_err(|e| format!("opening source shard {}: {e}",shard.path.display()))?;
            if reader.file_size() != shard.file_size_bytes {
                return Err(format!("source GGUF shard changed size: {}",shard.path.display()));
            }
            sources.push(Mutex::new(reader));
        }
        let mut by_tensor_name = HashMap::new();
        for (index, tensor) in inventory.expert_sources.iter().enumerate() {
            if by_tensor_name.insert(tensor.tensor_name.clone(),index).is_some() {
                return Err(format!("duplicate expert tensor {}",tensor.tensor_name));
            }
            let source = inventory.shards.get((tensor.shard_number as usize).saturating_sub(1))
                .ok_or("expert tensor references a nonexistent shard")?;
            if tensor.shard_number == 0 || source.shard_number != tensor.shard_number ||
               tensor.source_path != source.path {
                return Err("expert tensor source does not match validated shard".into());
            }
            let end = tensor.source_byte_offset.checked_add(tensor.byte_length)
                .ok_or("expert source offset overflow")?;
            if tensor.byte_length == 0 || end > source.file_size_bytes {
                return Err(format!("expert tensor {} exceeds its source GGUF",
                    tensor.tensor_name));
            }
        }
        Ok(Self {inventory,sources,by_tensor_name})
    }

    pub fn inventory(&self) -> &SplitGgufInventory { &self.inventory }

    pub fn source_for(&self, tensor_name: &str) -> Option<&SplitExpertSource> {
        self.by_tensor_name.get(tensor_name)
            .and_then(|index| self.inventory.expert_sources.get(*index))
    }

    /// Read a checked tensor span into a caller-owned buffer. This avoids
    /// allocating full model files and allows safe per-source async planning.
    pub fn read_tensor_into(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
        let tensor = self.source_for(name)
            .ok_or_else(|| format!("expert tensor {name} not indexed"))?;
        if destination.len() as u64 != tensor.byte_length {
            return Err(format!("expert tensor {name} needs exactly {} bytes, got {}",
                tensor.byte_length,destination.len()));
        }
        let reader = self.sources.get(tensor.shard_number as usize - 1)
            .ok_or("expert source shard is not open")?;
        reader.lock().unwrap_or_else(|e| e.into_inner())
            .read_exact_at(tensor.source_byte_offset,destination)
            .map_err(|e| format!("expert tensor {name} source read failed: {e}"))
    }

    /// Read one native-layout fused expert slab without first materializing
    /// the full fused projection. This is the exact per-expert I/O primitive
    /// needed for a future native graph callback with per-shard descriptors.
    pub fn read_fused_expert_into(
        &self, layer: u32, expert_id: u32, projection_name: &str,
        destination: &mut [u8],
    ) -> Result<(), String> {
        let slab = self.inventory.fused_expert_slabs.iter().find(|entry| {
            entry.layer == layer && entry.expert_id == expert_id &&
                entry.tensor_name == projection_name
        }).ok_or_else(|| format!(
            "fused expert {layer}:{expert_id} projection {projection_name} is not in a complete verified multi-shard map"
        ))?;
        if destination.len() as u64 != slab.byte_length {
            return Err(format!(
                "expert slab needs exactly {} bytes, got {}",
                slab.byte_length,destination.len()));
        }
        let source = self.sources.get(slab.shard_number as usize - 1)
            .ok_or("fused expert slab references a missing source file")?;
        source.lock().unwrap_or_else(|e| e.into_inner())
            .read_exact_at(slab.source_byte_offset,destination)
            .map_err(|e| format!("fused expert {layer}:{expert_id} source read failed: {e}"))
    }

    /// Clone pinned file descriptors for parallel expert reads without
    /// reopening an attacker-replaceable filesystem pathname.
    pub fn fork_source_readers(&self) -> Result<Vec<FileRangeReader>, String> {
        self.sources.iter().map(|reader| {
            reader.lock().unwrap_or_else(|e| e.into_inner())
                .try_clone().map_err(|e| format!("cloning split source descriptor: {e}"))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_models::split_gguf::{SplitShardSummary, SplitExpertSource};

    #[test]
    fn reads_exact_ranges_from_each_shard_without_concatenation() {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("kiln-source-{}-{nonce}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let a=dir.join("a.gguf");
        let b=dir.join("b.gguf");
        std::fs::write(&a,(0u8..100).collect::<Vec<_>>()).unwrap();
        std::fs::write(&b,(100u8..200).collect::<Vec<_>>()).unwrap();
        let inventory=SplitGgufInventory {
            entrypoint:a.clone(),shard_count:2,total_file_bytes:200,
            total_tensor_count:2,architecture:Some("tiny".into()),
            shards:vec![
                SplitShardSummary{shard_number:1,path:a.clone(),file_size_bytes:100,
                    tensor_count:1,expert_tensor_count:1,unsupported_expert_tensor_count:0},
                SplitShardSummary{shard_number:2,path:b.clone(),file_size_bytes:100,
                    tensor_count:1,expert_tensor_count:1,unsupported_expert_tensor_count:0},
            ],
            expert_sources:vec![
                SplitExpertSource{tensor_name:"up".into(),shard_number:1,
                    source_path:a.clone(),source_byte_offset:20,byte_length:10},
                SplitExpertSource{tensor_name:"down".into(),shard_number:2,
                    source_path:b.clone(),source_byte_offset:10,byte_length:8},
            ],
            fused_expert_slabs:vec![],unsupported_expert_tensors:vec![],native_multi_source_streaming_ready:false,
        };
        let reader=SplitGgufExpertReader::from_inventory(inventory).unwrap();
        let mut up=[0u8;10];let mut down=[0u8;8];
        reader.read_tensor_into("up",&mut up).unwrap();
        reader.read_tensor_into("down",&mut down).unwrap();
        assert_eq!(up.to_vec(),(20u8..30).collect::<Vec<_>>());
        assert_eq!(down.to_vec(),(110u8..118).collect::<Vec<_>>());
        assert!(reader.read_tensor_into("unknown",&mut up).is_err());
        assert!(reader.read_tensor_into("up",&mut down).is_err());
        let mut fork=reader.fork_source_readers().unwrap();
        assert_eq!(fork[1].read_range(10,8).unwrap(),down);
        std::fs::remove_file(&b).unwrap();
        // Existing descriptors remain valid if a model path disappears.
        reader.read_tensor_into("down",&mut down).unwrap();
        assert_eq!(down.to_vec(),(110u8..118).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_only_selected_fused_slab_from_correct_original_shard() {
        use kiln_models::split_gguf::SplitFusedExpertSlab;
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("kiln-fused-slab-{}-{nonce}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let first=dir.join("one.gguf");
        let second=dir.join("two.gguf");
        std::fs::write(&first,[0u8;128]).unwrap();
        let mut content=[0u8;128];
        content[32..40].copy_from_slice(b"EXPERT00");
        content[40..48].copy_from_slice(b"EXPERT01");
        std::fs::write(&second,content).unwrap();
        let inv=SplitGgufInventory {
            entrypoint:first.clone(),shard_count:2,total_file_bytes:256,
            total_tensor_count:1,architecture:Some("tiny".into()),
            shards:vec![
                SplitShardSummary{shard_number:1,path:first.clone(),file_size_bytes:128,
                    tensor_count:0,expert_tensor_count:0,unsupported_expert_tensor_count:0},
                SplitShardSummary{shard_number:2,path:second.clone(),file_size_bytes:128,
                    tensor_count:1,expert_tensor_count:1,unsupported_expert_tensor_count:0},
            ],
            expert_sources:vec![],
            fused_expert_slabs:(0..2).map(|expert_id|SplitFusedExpertSlab{
                tensor_name:"blk.0.ffn_up_exps.weight".into(),
                layer:0,expert_id,shard_number:2,source_path:second.clone(),
                source_byte_offset:32+u64::from(expert_id)*8,byte_length:8,
            }).collect(),
            unsupported_expert_tensors:vec![],native_multi_source_streaming_ready:false,
        };
        let reader=SplitGgufExpertReader::from_inventory(inv).unwrap();
        let mut out=[0u8;8];
        reader.read_fused_expert_into(0,1,"blk.0.ffn_up_exps.weight",&mut out).unwrap();
        assert_eq!(&out,b"EXPERT01");
        assert!(reader.read_fused_expert_into(0,2,"blk.0.ffn_up_exps.weight",&mut out).is_err());
        assert!(reader.read_fused_expert_into(0,0,"blk.0.ffn_up_exps.weight",&mut out[..7]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_out_of_bounds_and_duplicate_component_sources() {
        let file=std::env::temp_dir().join(format!("kiln-split-invalid-{}.dat",std::process::id()));
        std::fs::write(&file,[0u8;12]).unwrap();
        let shard=SplitShardSummary{shard_number:1,path:file.clone(),file_size_bytes:12,
            tensor_count:1,expert_tensor_count:1,unsupported_expert_tensor_count:0};
        let component=SplitExpertSource{tensor_name:"up".into(),shard_number:1,
            source_path:file.clone(),source_byte_offset:11,byte_length:2};
        let mut inv=SplitGgufInventory{entrypoint:file.clone(),shard_count:1,
            total_file_bytes:12,total_tensor_count:1,architecture:None,
            shards:vec![shard],expert_sources:vec![component.clone()],
            fused_expert_slabs:vec![],unsupported_expert_tensors:vec![],native_multi_source_streaming_ready:false};
        assert!(SplitGgufExpertReader::from_inventory(inv.clone()).is_err());
        inv.expert_sources[0].source_byte_offset=0;
        inv.expert_sources.push(inv.expert_sources[0].clone());
        assert!(SplitGgufExpertReader::from_inventory(inv).is_err());
        std::fs::remove_file(file).unwrap();
    }
}
