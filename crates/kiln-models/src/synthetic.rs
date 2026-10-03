//! Synthetic model format for pipeline testing.
//!
//! See docs/pipeline-design.md, section "What load a model means for
//! the first version". This is not a real model format. It exists so
//! the pipeline can be exercised end to end before GGUF loading lands.
//!
//! File layout:
//!   "KILN_SYNTHETIC_v1\n"
//!   "num_rows: <N>\n"
//!   "num_cols: <M>\n"
//!   "scale: <F>\n"
//!   "weights: " followed by ceil(N*M/5) bytes of TQ1.0 packed trits
//!   "activations: " followed by M bytes of int8 values
//!
//! The weights and activations sections are binary. The header is text.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Errors from reading or writing synthetic models.
#[derive(Debug)]
pub enum SyntheticError {
    Io(std::io::Error),
    MalformedHeader(String),
    TruncatedData { expected: usize, got: usize },
}

impl std::fmt::Display for SyntheticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyntheticError::Io(e) => write!(f, "io error: {}", e),
            SyntheticError::MalformedHeader(msg) => write!(f, "malformed header: {}", msg),
            SyntheticError::TruncatedData { expected, got } => {
                write!(f, "truncated data: expected {} bytes, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for SyntheticError {}

impl From<std::io::Error> for SyntheticError {
    fn from(e: std::io::Error) -> Self {
        SyntheticError::Io(e)
    }
}

/// A synthetic model in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticModel {
    pub num_rows: u32,
    pub num_cols: u32,
    pub scale: f32,
    /// Packed TQ1.0 weights. Length is ceil(num_rows * num_cols / 5).
    pub weights: Vec<u8>,
    /// int8 activations. Length is num_cols.
    pub activations: Vec<i8>,
}

impl SyntheticModel {
    /// The total number of ternary weights in the model.
    pub fn total_weights(&self) -> usize {
        self.num_rows as usize * self.num_cols as usize
    }

    /// The number of packed weight bytes.
    pub fn packed_weight_bytes(&self) -> usize {
        (self.total_weights() + 4) / 5
    }
}

/// Write a synthetic model to a file.
pub fn write_synthetic(path: &Path, model: &SyntheticModel) -> Result<(), SyntheticError> {
    let mut f = fs::File::create(path)?;

    write!(f, "KILN_SYNTHETIC_v1\n")?;
    write!(f, "num_rows: {}\n", model.num_rows)?;
    write!(f, "num_cols: {}\n", model.num_cols)?;
    write!(f, "scale: {}\n", model.scale)?;
    write!(f, "weights: ")?;
    f.write_all(&model.weights)?;
    write!(f, "\nactivations: ")?;
    // Write int8 bytes as u8.
    let acts_u8: Vec<u8> = model.activations.iter().map(|&x| x as u8).collect();
    f.write_all(&acts_u8)?;
    write!(f, "\n")?;

    Ok(())
}

/// Read a synthetic model from a file.
pub fn read_synthetic(path: &Path) -> Result<SyntheticModel, SyntheticError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;

    // Find the byte positions of the weights and activations markers.
    let weights_marker: &[u8] = b"weights: ";
    let acts_marker: &[u8] = b"\nactivations: ";

    let weights_pos = find_subslice(&bytes, weights_marker)
        .ok_or_else(|| SyntheticError::MalformedHeader("missing 'weights: ' marker".to_string()))?;
    let weights_start = weights_pos + weights_marker.len();

    let acts_pos = find_subslice(&bytes[weights_start..], acts_marker)
        .ok_or_else(|| SyntheticError::MalformedHeader("missing 'activations: ' marker".to_string()))?;
    let acts_start = weights_start + acts_pos + acts_marker.len();

    // Parse the header text (everything before weights_start).
    let header = std::str::from_utf8(&bytes[..weights_pos])
        .map_err(|_| SyntheticError::MalformedHeader("header is not valid UTF-8".to_string()))?;

    let mut lines = header.lines();
    let magic = lines.next().unwrap_or("");
    if magic != "KILN_SYNTHETIC_v1" {
        return Err(SyntheticError::MalformedHeader(format!(
            "bad magic: expected KILN_SYNTHETIC_v1, got {}",
            magic
        )));
    }

    let mut num_rows: Option<u32> = None;
    let mut num_cols: Option<u32> = None;
    let mut scale: Option<f32> = None;

    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| SyntheticError::MalformedHeader(format!("bad line: {}", line)))?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "num_rows" => {
                num_rows = Some(value.parse().map_err(|_| {
                    SyntheticError::MalformedHeader(format!("bad num_rows: {}", value))
                })?);
            }
            "num_cols" => {
                num_cols = Some(value.parse().map_err(|_| {
                    SyntheticError::MalformedHeader(format!("bad num_cols: {}", value))
                })?);
            }
            "scale" => {
                scale = Some(value.parse().map_err(|_| {
                    SyntheticError::MalformedHeader(format!("bad scale: {}", value))
                })?);
            }
            _ => {}
        }
    }

    let num_rows = num_rows.ok_or_else(|| {
        SyntheticError::MalformedHeader("missing num_rows".to_string())
    })?;
    let num_cols = num_cols.ok_or_else(|| {
        SyntheticError::MalformedHeader("missing num_cols".to_string())
    })?;
    let scale = scale.ok_or_else(|| {
        SyntheticError::MalformedHeader("missing scale".to_string())
    })?;

    // Compute expected sizes.
    let total_weights = num_rows as usize * num_cols as usize;
    let packed_bytes = (total_weights + 4) / 5;
    let acts_bytes = num_cols as usize;

    // The bytes between weights_start and acts_start - acts_marker.len() are
    // the weights. The acts_marker starts with a newline, so the weights end
    // at acts_pos - 1 (exclusive of the leading newline of the marker).
    let weights_end = weights_start + acts_pos;
    let weight_bytes_available = weights_end - weights_start;

    if weight_bytes_available < packed_bytes {
        return Err(SyntheticError::TruncatedData {
            expected: packed_bytes,
            got: weight_bytes_available,
        });
    }

    let weights = bytes[weights_start..weights_start + packed_bytes].to_vec();

    // Activations start after the marker.
    let acts_available = bytes.len() - acts_start;
    if acts_available < acts_bytes {
        return Err(SyntheticError::TruncatedData {
            expected: acts_bytes,
            got: acts_available,
        });
    }

    let activations: Vec<i8> = bytes[acts_start..acts_start + acts_bytes]
        .iter()
        .map(|&b| b as i8)
        .collect();

    Ok(SyntheticModel {
        num_rows,
        num_cols,
        scale,
        weights,
        activations,
    })
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kiln_synth_test_{}_{}.bin", name, std::process::id()));
        p
    }

    #[test]
    fn round_trip_small() {
        let model = SyntheticModel {
            num_rows: 4,
            num_cols: 10,
            scale: 1.5,
            weights: vec![0u8; 8], // ceil(40/5) = 8
            activations: vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };
        let path = tmp_path("small");
        write_synthetic(&path, &model).unwrap();
        let read_back = read_synthetic(&path).unwrap();
        assert_eq!(read_back.num_rows, 4);
        assert_eq!(read_back.num_cols, 10);
        assert_eq!(read_back.scale, 1.5);
        assert_eq!(read_back.weights, model.weights);
        assert_eq!(read_back.activations, model.activations);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn malformed_magic() {
        let path = tmp_path("bad_magic");
        std::fs::write(&path, b"NOT_KILN\nnum_rows: 1\n").unwrap();
        let result = read_synthetic(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncated_weights() {
        let path = tmp_path("trunc");
        // Claim 100 weights but only provide 1 byte.
        let mut data = Vec::new();
        data.extend_from_slice(b"KILN_SYNTHETIC_v1\nnum_rows: 10\nnum_cols: 10\nscale: 1.0\nweights: ");
        data.push(0u8);
        data.extend_from_slice(b"\nactivations: ");
        data.extend_from_slice(&[0u8; 10]);
        data.push(b'\n');
        std::fs::write(&path, &data).unwrap();
        let result = read_synthetic(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
}
