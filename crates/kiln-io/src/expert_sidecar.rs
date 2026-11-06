//! GGUF-preserving Phase 5.1 expert sidecar pack, with bounded streaming
//! writes, deterministic 4 KiB alignment, and per-expert CRC32 integrity.
//!
//! This cannot infer expert tensors by itself. Callers provide a validated
//! ExpertTable of byte ranges from the original, unchanged GGUF (or another
//! source blob). No GGUF header, tensor table or source byte is modified.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use kiln_models::{ExpertRange, ExpertTable};
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
    file.write_all(&bytes)?;
    file.sync_all()?;
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
        for range in &ranges.ranges {
            let start = aligned_position(position)?;
            let mut gap = start - position;
            while gap > 0 {
                let count = usize::try_from(gap.min(CHUNK as u64)).unwrap();
                output.write_all(&buf[..count])?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

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
}
