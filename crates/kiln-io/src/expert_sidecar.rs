//! GGUF-preserving Phase 5.1 expert sidecar pack, with bounded streaming
//! writes, deterministic 4 KiB alignment, and per-expert CRC32 integrity.
//!
//! This cannot infer expert tensors by itself. Callers provide a validated
//! ExpertTable of byte ranges from the original, unchanged GGUF (or another
//! source blob). No GGUF header, tensor table or source byte is modified.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use kiln_models::{ExpertRange, ExpertTable, GgufExpertSourceMap};
use serde::{Deserialize, Serialize};

pub const SIDECAR_ALIGNMENT: u64 = 4096;
const CHUNK: usize = 64 * 1024;
const MAX_MANIFEST_BYTES: u64 = 128 * 1024 * 1024;
const MAX_EXPERTS: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertIntegrity {
    pub layer: u32,
    pub expert_id: u32,
    pub crc32: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarIntegrityManifest {
    pub format_version: u32,
    pub alignment_bytes: u64,
    pub source_size_bytes: u64,
    pub experts: Vec<ExpertIntegrity>,
}

#[derive(Debug)]
pub enum SidecarPackError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}
impl std::fmt::Display for SidecarPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "sidecar I/O: {e}"),
            Self::Json(e) => write!(f, "sidecar JSON: {e}"),
            Self::Invalid(e) => write!(f, "invalid sidecar: {e}"),
        }
    }
}
impl std::error::Error for SidecarPackError {}
impl From<std::io::Error> for SidecarPackError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}
impl From<serde_json::Error> for SidecarPackError {
    fn from(e: serde_json::Error) -> Self { Self::Json(e) }
}

fn invalid(msg: impl Into<String>) -> SidecarPackError {
    SidecarPackError::Invalid(msg.into())
}

/// Standard IEEE CRC32, useful for corruption detection, not authentication.
fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 == 1 { 0xedb8_8320 } else { 0 };
        }
    }
    crc
}

fn aligned_position(position: u64) -> Result<u64, SidecarPackError> {
    let delta = (SIDECAR_ALIGNMENT - position % SIDECAR_ALIGNMENT) % SIDECAR_ALIGNMENT;
    position.checked_add(delta).ok_or_else(|| invalid("sidecar offset overflow"))
}

fn validate_distinct_paths(paths: &[&Path]) -> Result<(), SidecarPackError> {
    let mut unique = std::collections::HashSet::new();
    for path in paths {
        // Parent must exist; canonical parent catches alias paths and symlinks.
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let canonical_parent = std::fs::canonicalize(parent)?;
        let name = path.file_name().ok_or_else(|| invalid("output path has no filename"))?;
        let full = canonical_parent.join(name);
        if !unique.insert(full) { return Err(invalid("source and output paths must be distinct")); }
    }
    Ok(())
}

fn write_json_new<T: Serialize>(path: &Path, value: &T) -> Result<(), SidecarPackError> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(invalid("index or integrity manifest exceeds 128 MiB"));
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let result = file.write_all(&bytes).and_then(|_| file.sync_all());
    drop(file);
    if let Err(error) = result {
        // Only this call's create_new output is removed. Existing artifacts
        // are never overwritten or deleted on error.
        let _ = std::fs::remove_file(path);
        return Err(SidecarPackError::Io(error));
    }
    Ok(())
}

/// Copy only declared expert ranges from the source into an aligned sidecar.
/// Produces an existing-API-compatible ExpertTable index plus a parallel
/// integrity JSON. Never overwrites an existing output or mutates the source.
/// Any partial newly created outputs are removed when writing fails.
pub fn write_expert_sidecar(
    source: &Path,
    ranges: ExpertTable,
    blob_path: &Path,
    index_path: &Path,
    integrity_path: &Path,
) -> Result<SidecarIntegrityManifest, SidecarPackError> {
    validate_distinct_paths(&[source, blob_path, index_path, integrity_path])?;
    let mut input = File::open(source)?;
    let source_size = input.metadata()?.len();
    ranges.validate(source_size).map_err(|e| invalid(e.to_string()))?;
    if ranges.ranges.is_empty() || ranges.ranges.len() > MAX_EXPERTS {
        return Err(invalid("expert count must be 1..=1,000,000"));
    }
    // Fail before writing any bytes if the metadata cannot fit the format.
    let max_data = ranges.ranges.iter().try_fold(0u64, |sum, entry| {
        let start = aligned_position(sum).ok()?;
        start.checked_add(entry.byte_length)
    }).ok_or_else(|| invalid("expert sidecar size overflows u64"))?;
    if max_data > i64::MAX as u64 {
        return Err(invalid("expert sidecar cannot exceed signed file offset limit"));
    }

    let mut created: Vec<PathBuf> = Vec::new();
    let result = (|| -> Result<_, SidecarPackError> {
        let mut output = OpenOptions::new().write(true).create_new(true).open(blob_path)?;
        created.push(blob_path.to_owned());
        let mut position = 0u64;
        let mut result_ranges = Vec::with_capacity(ranges.ranges.len());
        let mut checksums = Vec::with_capacity(ranges.ranges.len());
        let mut buf = [0u8; CHUNK];
        let zeros = [0u8; CHUNK];
        for range in &ranges.ranges {
            let start = aligned_position(position)?;
            let mut gap = start - position;
            while gap > 0 {
                let count = usize::try_from(gap.min(CHUNK as u64)).unwrap();
                output.write_all(&zeros[..count])?;
                gap -= count as u64;
            }
            input.seek(SeekFrom::Start(range.byte_offset))?;
            let mut bytes_left = range.byte_length;
            let mut crc = !0u32;
            while bytes_left > 0 {
                let count = usize::try_from(bytes_left.min(CHUNK as u64)).unwrap();
                input.read_exact(&mut buf[..count])?;
                output.write_all(&buf[..count])?;
                crc = crc32_update(crc, &buf[..count]);
                bytes_left -= count as u64;
            }
            result_ranges.push(ExpertRange {
                layer: range.layer, expert_id: range.expert_id,
                byte_offset: start, byte_length: range.byte_length,
            });
            checksums.push(ExpertIntegrity {
                layer: range.layer, expert_id: range.expert_id, crc32: !crc,
            });
            position = start + range.byte_length;
        }
        output.sync_all()?;
        let table = ExpertTable {
            blob_version: 1, blob_size: position, ranges: result_ranges,
        };
        table.validate(position).map_err(|e| invalid(e.to_string()))?;
        let manifest = SidecarIntegrityManifest {
            format_version: 1,
            alignment_bytes: SIDECAR_ALIGNMENT,
            source_size_bytes: source_size,
            experts: checksums,
        };
        write_json_new(index_path, &table)?;
        created.push(index_path.to_owned());
        write_json_new(integrity_path, &manifest)?;
        created.push(integrity_path.to_owned());
        Ok(manifest)
    })();
    if result.is_err() {
        for path in created.iter().rev() { let _ = std::fs::remove_file(path); }
    }
    result
}

fn read_manifest_bounded(path: &Path) -> Result<Vec<u8>, SidecarPackError> {
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_MANIFEST_BYTES {
        return Err(invalid("index or integrity manifest exceeds 128 MiB"));
    }
    let mut data = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut data)?;
    if data.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(invalid("metadata read exceeds 128 MiB"));
    }
    Ok(data)
}

/// Verify every expert's CRC32 independently against the packed bytes.
/// Reject bad alignment, malformed indices, changed lengths or corrupt data.
pub fn verify_expert_sidecar(
    blob_path: &Path, index_path: &Path, integrity_path: &Path,
) -> Result<ExpertTable, SidecarPackError> {
    let index: ExpertTable = serde_json::from_slice(&read_manifest_bounded(index_path)?)?;
    let integrity: SidecarIntegrityManifest =
        serde_json::from_slice(&read_manifest_bounded(integrity_path)?)?;
    if integrity.format_version != 1 || integrity.alignment_bytes != SIDECAR_ALIGNMENT {
        return Err(invalid("unsupported integrity manifest format/alignment"));
    }
    let mut input = File::open(blob_path)?;
    let size = input.metadata()?.len();
    index.validate(size).map_err(|e| invalid(e.to_string()))?;
    if index.ranges.len() != integrity.experts.len()
        || index.ranges.is_empty() || index.ranges.len() > MAX_EXPERTS {
        return Err(invalid("index/integrity expert count mismatch"));
    }
    let mut buf = [0u8; CHUNK];
    for (range, check) in index.ranges.iter().zip(&integrity.experts) {
        if range.layer != check.layer || range.expert_id != check.expert_id
            || range.byte_offset % SIDECAR_ALIGNMENT != 0 {
            return Err(invalid("expert key mismatch or incorrect alignment"));
        }
        input.seek(SeekFrom::Start(range.byte_offset))?;
        let mut left = range.byte_length;
        let mut crc = !0u32;
        while left > 0 {
            let count = usize::try_from(left.min(CHUNK as u64)).unwrap();
            input.read_exact(&mut buf[..count])?;
            crc = crc32_update(crc, &buf[..count]);
            left -= count as u64;
        }
        if !crc != check.crc32 {
            return Err(invalid(format!(
                "CRC32 mismatch for expert ({}, {})", range.layer, range.expert_id
            )));
        }
    }
    Ok(index)
}

/// The location of one exact original GGUF tensor within the packed expert.
/// Component offsets are absolute within the sidecar blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedExpertComponent {
    pub layer: u32,
    pub expert_id: u32,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedExpertComponentMap {
    pub format_version: u32,
    pub components: Vec<PackedExpertComponent>,
}

/// Accept explicitly resolved GGUF tensor *components* for each expert and
/// concatenate them into an expert-contiguous sidecar, retaining a component
/// table so a later loader can identify the original tensor boundaries.
/// Does not alter the original source or attempt to write a stripped GGUF.
pub fn write_component_expert_sidecar(
    source: &Path,
    map: GgufExpertSourceMap,
    blob_path: &Path,
    index_path: &Path,
    integrity_path: &Path,
    component_map_path: &Path,
) -> Result<(SidecarIntegrityManifest, PackedExpertComponentMap), SidecarPackError> {
    validate_distinct_paths(&[
        source, blob_path, index_path, integrity_path, component_map_path
    ])?;
    let mut input = File::open(source)?;
    let source_size = input.metadata()?.len();
    if map.format_version != 1 || map.source_size_bytes != source_size
        || map.components.is_empty() || map.components.len() > MAX_EXPERTS {
        return Err(invalid("component map version, source size or count is invalid"));
    }
    let mut previous_key: Option<(u32, u32, &str)> = None;
    let mut spans = Vec::with_capacity(map.components.len());
    for item in &map.components {
        let key = (item.layer, item.expert_id, item.tensor_name.as_str());
        if previous_key.is_some_and(|prior| key <= prior) {
            return Err(invalid("component keys must be strictly sorted"));
        }
        previous_key = Some(key);
        if item.byte_length == 0 ||
            item.source_byte_offset.checked_add(item.byte_length)
                .is_none_or(|end| end > source_size) {
            return Err(invalid("component range out of source bounds"));
        }
        spans.push((item.source_byte_offset, item.byte_length));
    }
    spans.sort_unstable_by_key(|(offset, _)| *offset);
    for window in spans.windows(2) {
        if window[0].0 + window[0].1 > window[1].0 {
            return Err(invalid("component source byte ranges overlap"));
        }
    }

    // Validate every packed output offset before opening a destination.
    let mut position = 0u64;
    let mut last_expert: Option<(u32,u32)> = None;
    for item in &map.components {
        let key=(item.layer,item.expert_id);
        if last_expert != Some(key) {
            position=aligned_position(position)?;
            last_expert=Some(key);
        }
        position=position.checked_add(item.byte_length)
            .ok_or_else(|| invalid("packed component offset overflow"))?;
        if position > i64::MAX as u64 {
            return Err(invalid("sidecar exceeds supported file offset"));
        }
    }

    let mut created: Vec<PathBuf> = Vec::new();
    let result = (|| -> Result<_, SidecarPackError> {
        let mut output = OpenOptions::new().write(true).create_new(true).open(blob_path)?;
        created.push(blob_path.to_owned());
        let mut buffer = [0u8; CHUNK];
        let zeros = [0u8; CHUNK];
        let mut offset = 0u64;
        let mut packed_ranges: Vec<ExpertRange> = Vec::new();
        let mut checksums: Vec<ExpertIntegrity> = Vec::new();
        let mut components: Vec<PackedExpertComponent> = Vec::with_capacity(map.components.len());
        let mut current_key: Option<(u32,u32)> = None;
        let mut start = 0u64;
        let mut crc = !0u32;
        for item in &map.components {
            let key=(item.layer,item.expert_id);
            if current_key != Some(key) {
                if let Some((layer,expert_id))=current_key {
                    packed_ranges.push(ExpertRange {
                        layer,expert_id,byte_offset:start,byte_length:offset-start,
                    });
                    checksums.push(ExpertIntegrity {layer,expert_id,crc32:!crc});
                }
                let aligned=aligned_position(offset)?;
                let mut gap=aligned-offset;
                while gap>0 {
                    let size=usize::try_from(gap.min(CHUNK as u64)).unwrap();
                    output.write_all(&zeros[..size])?;
                    gap-=size as u64;
                }
                offset=aligned;
                start=offset;
                crc=!0u32;
                current_key=Some(key);
            }
            components.push(PackedExpertComponent {
                layer:item.layer,expert_id:item.expert_id,
                tensor_name:item.tensor_name.clone(),
                byte_offset:offset,byte_length:item.byte_length,
            });
            input.seek(SeekFrom::Start(item.source_byte_offset))?;
            let mut left=item.byte_length;
            while left>0 {
                let n=usize::try_from(left.min(CHUNK as u64)).unwrap();
                input.read_exact(&mut buffer[..n])?;
                output.write_all(&buffer[..n])?;
                crc=crc32_update(crc,&buffer[..n]);
                offset+=n as u64;
                left-=n as u64;
            }
        }
        if let Some((layer,expert_id))=current_key {
            packed_ranges.push(ExpertRange {
                layer,expert_id,byte_offset:start,byte_length:offset-start,
            });
            checksums.push(ExpertIntegrity {layer,expert_id,crc32:!crc});
        }
        output.sync_all()?;
        let index=ExpertTable {
            blob_version:1,blob_size:offset,ranges:packed_ranges,
        };
        index.validate(offset).map_err(|e| invalid(e.to_string()))?;
        let integrity=SidecarIntegrityManifest {
            format_version:1,alignment_bytes:SIDECAR_ALIGNMENT,
            source_size_bytes:source_size,experts:checksums,
        };
        let map=PackedExpertComponentMap {
            format_version:1,components,
        };
        write_json_new(index_path,&index)?;
        created.push(index_path.to_owned());
        write_json_new(integrity_path,&integrity)?;
        created.push(integrity_path.to_owned());
        write_json_new(component_map_path,&map)?;
        created.push(component_map_path.to_owned());
        Ok((integrity,map))
    })();
    if result.is_err() {
        for file in created.iter().rev() { let _ = std::fs::remove_file(file); }
    }
    result
}

/// Verify sidecar payloads AND the component map. Every expert's components
/// must cover its byte range exactly, with no omissions, reordering or gaps.
pub fn verify_component_expert_sidecar(
    blob_path: &Path, index_path: &Path, integrity_path: &Path,
    component_map_path: &Path,
) -> Result<(ExpertTable,PackedExpertComponentMap),SidecarPackError> {
    let table=verify_expert_sidecar(blob_path,index_path,integrity_path)?;
    let map: PackedExpertComponentMap =
        serde_json::from_slice(&read_manifest_bounded(component_map_path)?)?;
    if map.format_version != 1 || map.components.is_empty()
        || map.components.len()>MAX_EXPERTS {
        return Err(invalid("component manifest version or count invalid"));
    }
    let mut cursor=0usize;
    for expert in &table.ranges {
        let mut position=expert.byte_offset;
        let mut has_component=false;
        while let Some(component)=map.components.get(cursor) {
            if (component.layer,component.expert_id)
                != (expert.layer,expert.expert_id) { break; }
            if component.byte_length == 0 || component.byte_offset != position {
                return Err(invalid("component has gap, overlap or zero length"));
            }
            position=position.checked_add(component.byte_length)
                .ok_or_else(|| invalid("component offset overflow"))?;
            if position > expert.byte_offset + expert.byte_length {
                return Err(invalid("component exceeds expert range"));
            }
            if cursor > 0 {
                let previous=&map.components[cursor-1];
                if (previous.layer,previous.expert_id)==(component.layer,component.expert_id)
                    && previous.tensor_name >= component.tensor_name {
                    return Err(invalid("component tensor names are not sorted"));
                }
            }
            cursor+=1;
            has_component=true;
        }
        if !has_component || position != expert.byte_offset+expert.byte_length {
            return Err(invalid("component map does not cover entire expert"));
        }
    }
    if cursor != map.components.len() {
        return Err(invalid("component map contains unindexed experts"));
    }
    Ok((table,map))
}


/// End-to-end source compatibility proof for one explicit component map.
/// Streams both files in fixed-size chunks; never loads a complete expert or
/// GGUF into RAM. This is stronger than the packed CRC32 alone because it
/// verifies that each named packed component matches the original source.
pub fn verify_component_sidecar_against_source(
    source: &Path,
    source_map: &GgufExpertSourceMap,
    blob_path: &Path,
    index_path: &Path,
    integrity_path: &Path,
    component_map_path: &Path,
) -> Result<usize, SidecarPackError> {
    let (table, packed) = verify_component_expert_sidecar(
        blob_path, index_path, integrity_path, component_map_path,
    )?;
    let mut original = File::open(source)?;
    let size = original.metadata()?.len();
    if source_map.format_version != 1 || source_map.source_size_bytes != size
        || source_map.components.len() != packed.components.len() {
        return Err(invalid("source mapping version, file size or component count differs"));
    }
    if table.blob_size == 0 {
        return Err(invalid("expert sidecar is empty"));
    }
    let mut sidecar = File::open(blob_path)?;
    let mut source_buf = [0u8; CHUNK];
    let mut packed_buf = [0u8; CHUNK];
    for (src, dst) in source_map.components.iter().zip(&packed.components) {
        if src.layer != dst.layer || src.expert_id != dst.expert_id
            || src.tensor_name != dst.tensor_name || src.byte_length != dst.byte_length {
            return Err(invalid("component tensor identity/size differs from the source mapping"));
        }
        if src.source_byte_offset.checked_add(src.byte_length)
            .is_none_or(|end| end > size) {
            return Err(invalid("source component exceeds source file"));
        }
        original.seek(SeekFrom::Start(src.source_byte_offset))?;
        sidecar.seek(SeekFrom::Start(dst.byte_offset))?;
        let mut remaining = src.byte_length;
        while remaining > 0 {
            let take = usize::try_from(remaining.min(CHUNK as u64)).unwrap();
            original.read_exact(&mut source_buf[..take])?;
            sidecar.read_exact(&mut packed_buf[..take])?;
            if source_buf[..take] != packed_buf[..take] {
                return Err(invalid(format!(
                    "packed tensor {} in expert ({}, {}) differs from original source",
                    src.tensor_name,src.layer,src.expert_id
                )));
            }
            remaining -= take as u64;
        }
    }
    Ok(packed.components.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

    #[test]
    fn packs_quantized_fused_experts_byte_exact_with_source_verification() {
        use kiln_models::{FusedExpertTensorSelection, GgufTensorInfo, GgufType,
            resolve_fused_expert_tensor_components};
        let root = std::env::temp_dir().join(format!(
            "kiln-fused-slab-parity-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("fused-source.bin");
        let blob=root.join("fused-experts.bin");
        let index=root.join("fused-index.json");
        let integrity=root.join("fused-integrity.json");
        let components=root.join("fused-components.json");
        let bytes:Vec<u8>=(0..8192).map(|i| ((i * 31 + 7) % 256) as u8).collect();
        std::fs::write(&source,&bytes).unwrap();
        let tensors=[
            GgufTensorInfo {
                name:"blk.0.ffn_up_exps".into(),
                shape:vec![256,3,4],dtype:GgufType::Q4_K,offset:0,
            },
            GgufTensorInfo {
                name:"blk.0.ffn_down_exps".into(),
                shape:vec![256,2,4],dtype:GgufType::Q4_K,offset:2048,
            },
        ];
        let map=resolve_fused_expert_tensor_components(
            &tensors,0,bytes.len() as u64,&[
                FusedExpertTensorSelection {
                    layer:0,tensor_name:"blk.0.ffn_up_exps".into(),expert_count:4,
                },
                FusedExpertTensorSelection {
                    layer:0,tensor_name:"blk.0.ffn_down_exps".into(),expert_count:4,
                },
            ],
        ).unwrap();
        let (checks,_)=write_component_expert_sidecar(
            &source,map.clone(),&blob,&index,&integrity,&components
        ).unwrap();
        assert_eq!(checks.experts.len(),4);
        let (table,packed)=verify_component_expert_sidecar(
            &blob,&index,&integrity,&components
        ).unwrap();
        assert_eq!(table.ranges.len(),4);
        assert_eq!(packed.components.len(),8);
        assert_eq!(verify_component_sidecar_against_source(
            &source,&map,&blob,&index,&integrity,&components
        ).unwrap(),8);
        // Mutating source detects content drift even when packed CRC32 is intact.
        let mut corrupted=bytes;
        corrupted[500]^=0xff;
        std::fs::write(&source,&corrupted).unwrap();
        assert!(verify_component_sidecar_against_source(
            &source,&map,&blob,&index,&integrity,&components
        ).is_err());
        for p in [&source,&blob,&index,&integrity,&components] {
            std::fs::remove_file(p).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn packed_components_match_source_and_detect_changed_source() {
        let root = std::env::temp_dir().join(format!(
            "kiln-sidecar-parity-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("source.bin");
        let blob=root.join("packed.bin");
        let index=root.join("index.json");
        let integrity=root.join("integrity.json");
        let manifest=root.join("components.json");
        let bytes:Vec<u8>=(0..=255u8).collect();
        std::fs::write(&source,&bytes).unwrap();
        let map=GgufExpertSourceMap {
            format_version:1,source_size_bytes:256,
            components:vec![
                kiln_models::ExpertTensorComponent {
                    layer:0,expert_id:0,tensor_name:"down".into(),
                    source_byte_offset:7,byte_length:51,
                },
                kiln_models::ExpertTensorComponent {
                    layer:0,expert_id:0,tensor_name:"up".into(),
                    source_byte_offset:80,byte_length:40,
                },
            ],
        };
        write_component_expert_sidecar(
            &source,map.clone(),&blob,&index,&integrity,&manifest,
        ).unwrap();
        assert_eq!(verify_component_sidecar_against_source(
            &source,&map,&blob,&index,&integrity,&manifest,
        ).unwrap(),2);
        let mut corrupted=bytes;
        corrupted[88]^=0xff;
        std::fs::write(&source,&corrupted).unwrap();
        assert!(verify_component_sidecar_against_source(
            &source,&map,&blob,&index,&integrity,&manifest,
        ).is_err());
        for p in [&source,&blob,&index,&integrity,&manifest] {
            std::fs::remove_file(p).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn known_crc32_vector() {
        assert_eq!(!crc32_update(!0, b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn deterministic_roundtrip_and_single_byte_corruption_detection() {
        let root = std::env::temp_dir().join(format!(
            "kiln-sidecar-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("source.gguf");
        let packed = root.join("experts.bin");
        let index = root.join("experts.index.json");
        let integrity = root.join("experts.crc32.json");
        let source_bytes: Vec<u8> = (0..255u8).collect();
        std::fs::write(&src, &source_bytes).unwrap();
        let source_table = ExpertTable {
            blob_version: 1, blob_size: source_bytes.len() as u64,
            ranges: vec![
                ExpertRange { layer: 0, expert_id: 0, byte_offset: 10, byte_length: 30 },
                ExpertRange { layer: 0, expert_id: 1, byte_offset: 70, byte_length: 75 },
                ExpertRange { layer: 1, expert_id: 0, byte_offset: 180, byte_length: 50 },
            ],
        };
        write_expert_sidecar(&src, source_table.clone(), &packed, &index, &integrity).unwrap();
        assert_eq!(std::fs::read(&src).unwrap(), source_bytes);
        let verified = verify_expert_sidecar(&packed, &index, &integrity).unwrap();
        assert_eq!(verified.ranges[1].byte_offset, 4096);
        assert_eq!(verified.ranges[2].byte_offset, 8192);
        let packed_bytes = std::fs::read(&packed).unwrap();
        assert!(packed_bytes[30..4096].iter().all(|byte| *byte == 0),
            "alignment gaps must be zero-filled, not previous expert data");
        assert!(packed_bytes[4096 + 75..8192].iter().all(|byte| *byte == 0));
        let mut reader = File::open(&packed).unwrap();
        for (src_range, packed_range) in source_table.ranges.iter().zip(verified.ranges.iter()) {
            let mut data = vec![0u8; packed_range.byte_length as usize];
            reader.seek(SeekFrom::Start(packed_range.byte_offset)).unwrap();
            reader.read_exact(&mut data).unwrap();
            assert_eq!(data, source_bytes[src_range.byte_offset as usize
                ..(src_range.byte_offset + src_range.byte_length) as usize]);
        }
        let mut file = OpenOptions::new().write(true).open(&packed).unwrap();
        file.seek(SeekFrom::Start(4096)).unwrap();
        file.write_all(&[0xab]).unwrap();
        assert!(verify_expert_sidecar(&packed, &index, &integrity).is_err());
        for p in [&src,&packed,&index,&integrity] { std::fs::remove_file(p).unwrap(); }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn invalid_ranges_fail_before_creating_any_outputs() {
        let root = std::env::temp_dir().join(format!(
            "kiln-bad-sidecar-{}-{:?}", std::process::id(), std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("source.bin");
        let dst = root.join("blob.bin");
        let idx = root.join("index.json");
        let crc = root.join("crc.json");
        std::fs::write(&src, vec![1u8; 16]).unwrap();
        let table = ExpertTable {blob_version:1,blob_size:16,ranges:vec![
            ExpertRange {layer:0,expert_id:0,byte_offset:12,byte_length:8}
        ]};
        assert!(write_expert_sidecar(&src,table,&dst,&idx,&crc).is_err());
        assert!(!dst.exists() && !idx.exists() && !crc.exists());
        assert!(write_expert_sidecar(&src,ExpertTable {
            blob_version:1,blob_size:16,ranges:vec![
                ExpertRange {layer:0,expert_id:0,byte_offset:0,byte_length:4}
            ]
        },&src,&idx,&crc).is_err());
        std::fs::remove_file(src).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn cannot_overwrite_existing_artifact() {
        let root = std::env::temp_dir().join(format!(
            "kiln-no-overwrite-{}-{:?}", std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let src=root.join("source");
        let dst=root.join("destination");
        let index=root.join("index");
        let integrity=root.join("integrity");
        std::fs::write(&src,[4u8;16]).unwrap();
        std::fs::write(&dst,b"preserve").unwrap();
        let table=ExpertTable{blob_version:1,blob_size:16,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:16}
        ]};
        assert!(write_expert_sidecar(&src,table,&dst,&index,&integrity).is_err());
        assert_eq!(std::fs::read(&dst).unwrap(),b"preserve");
        for p in [&src,&dst] {std::fs::remove_file(p).unwrap();}
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn multi_component_experts_roundtrip_exact_tensors_and_detect_map_tamper() {
        use kiln_models::{ExpertTensorComponent, GgufExpertSourceMap};
        let root = std::env::temp_dir().join(format!(
            "kiln-component-pack-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let src=root.join("source.gguf");
        let blob=root.join("experts.bin");
        let index=root.join("experts.index.json");
        let integrity=root.join("experts.crc32.json");
        let components=root.join("experts.components.json");
        let source_bytes:Vec<u8>=(0..=255u8).collect();
        std::fs::write(&src,&source_bytes).unwrap();
        let mapping=GgufExpertSourceMap {
            format_version:1,source_size_bytes:256,
            components:vec![
                ExpertTensorComponent {
                    layer:0,expert_id:0,tensor_name:"blk.0.ffn_down.0".into(),
                    source_byte_offset:30,byte_length:11,
                },
                ExpertTensorComponent {
                    layer:0,expert_id:0,tensor_name:"blk.0.ffn_gate.0".into(),
                    source_byte_offset:90,byte_length:13,
                },
                ExpertTensorComponent {
                    layer:0,expert_id:1,tensor_name:"blk.0.ffn_up.1".into(),
                    source_byte_offset:150,byte_length:17,
                },
            ],
        };
        let (_checksum,layout)=write_component_expert_sidecar(
            &src,mapping.clone(),&blob,&index,&integrity,&components
        ).unwrap();
        let (verified,verified_layout)=verify_component_expert_sidecar(
            &blob,&index,&integrity,&components
        ).unwrap();
        assert_eq!(verified.ranges.len(),2);
        assert_eq!(verified.ranges[0].byte_length,24);
        assert_eq!(verified.ranges[1].byte_offset,4096);
        assert_eq!(layout.components.len(),verified_layout.components.len());
        let packed=std::fs::read(&blob).unwrap();
        assert_eq!(&packed[0..11],&source_bytes[30..41]);
        assert_eq!(&packed[11..24],&source_bytes[90..103]);
        assert_eq!(&packed[4096..4113],&source_bytes[150..167]);
        assert!(packed[24..4096].iter().all(|b|*b==0));
        assert_eq!(std::fs::read(&src).unwrap(),source_bytes);

        let mut changed=verified_layout;
        changed.components[1].byte_offset+=1;
        std::fs::write(&components,serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(verify_component_expert_sidecar(
            &blob,&index,&integrity,&components
        ).is_err());
        for file in [&src,&blob,&index,&integrity,&components] {
            std::fs::remove_file(file).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn component_pack_rejects_overlapping_source_ranges_before_writing() {
        use kiln_models::{ExpertTensorComponent,GgufExpertSourceMap};
        let root=std::env::temp_dir().join(format!(
            "kiln-component-bad-{}-{:?}",std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let src=root.join("src");
        let blob=root.join("blob");
        let index=root.join("index");
        let integrity=root.join("integrity");
        let parts=root.join("parts");
        std::fs::write(&src,[7u8;100]).unwrap();
        let map=GgufExpertSourceMap{
            format_version:1,source_size_bytes:100,
            components:vec![
                ExpertTensorComponent{
                    layer:0,expert_id:0,tensor_name:"a".into(),
                    source_byte_offset:10,byte_length:20
                },
                ExpertTensorComponent{
                    layer:0,expert_id:0,tensor_name:"b".into(),
                    source_byte_offset:25,byte_length:10
                }
            ],
        };
        assert!(write_component_expert_sidecar(
            &src,map,&blob,&index,&integrity,&parts
        ).is_err());
        assert!(!blob.exists()&&!index.exists()&&!integrity.exists()&&!parts.exists());
        std::fs::remove_file(src).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

}
