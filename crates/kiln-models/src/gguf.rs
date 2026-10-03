//! Partial GGUF loader.
//!
//! See docs/gguf-design.md for the full design. This first version
//! implements the header parser. Metadata, tensor table, and data
//! section access land in subsequent commits.

use std::fs;
use std::io::Read;
use std::path::Path;

/// The GGUF magic number: "GGUF" as a little-endian u32.
pub const GGUF_MAGIC: u32 = 0x46554747;

/// The GGUF versions KILN supports.
pub const GGUF_VERSION_3: u32 = 3;

/// Errors from the GGUF loader.
#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    BadMagic { expected: u32, got: u32 },
    UnsupportedVersion(u32),
    TruncatedFile { needed: usize, got: usize },
    Metadata(String),
    Tensor(String),
    Alignment(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {}", e),
            GgufError::BadMagic { expected, got } => {
                write!(f, "bad magic: expected {:#x}, got {:#x}", expected, got)
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version: {}", v)
            }
            GgufError::TruncatedFile { needed, got } => {
                write!(f, "truncated file: needed {} bytes, got {}", needed, got)
            }
            GgufError::Metadata(msg) => write!(f, "metadata error: {}", msg),
            GgufError::Tensor(msg) => write!(f, "tensor error: {}", msg),
            GgufError::Alignment(msg) => write!(f, "alignment error: {}", msg),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// The GGUF header. The first 24 bytes of every GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// The byte size of the header.
    pub const SIZE: usize = 4 + 4 + 8 + 8;

    /// Parse a header from the first 24 bytes of a file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GgufError> {
        if bytes.len() < Self::SIZE {
            return Err(GgufError::TruncatedFile {
                needed: Self::SIZE,
                got: bytes.len(),
            });
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic {
                expected: GGUF_MAGIC,
                got: magic,
            });
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != GGUF_VERSION_3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let metadata_kv_count = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

/// Read the header from a GGUF file.
pub fn read_header(path: &Path) -> Result<GgufHeader, GgufError> {
    let mut f = fs::File::open(path)?;
    let mut buf = [0u8; GgufHeader::SIZE];
    f.read_exact(&mut buf)?;
    GgufHeader::from_bytes(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header_bytes(magic: u32, version: u32, tc: u64, mc: u64) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&magic.to_le_bytes());
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        buf[8..16].copy_from_slice(&tc.to_le_bytes());
        buf[16..24].copy_from_slice(&mc.to_le_bytes());
        buf
    }

    #[test]
    fn parse_valid_header() {
        let buf = make_header_bytes(GGUF_MAGIC, 3, 100, 25);
        let h = GgufHeader::from_bytes(&buf).unwrap();
        assert_eq!(h.magic, GGUF_MAGIC);
        assert_eq!(h.version, 3);
        assert_eq!(h.tensor_count, 100);
        assert_eq!(h.metadata_kv_count, 25);
    }

    #[test]
    fn reject_bad_magic() {
        let buf = make_header_bytes(0xDEADBEEF, 3, 0, 0);
        let result = GgufHeader::from_bytes(&buf);
        assert!(matches!(result, Err(GgufError::BadMagic { .. })));
    }

    #[test]
    fn reject_bad_version() {
        let buf = make_header_bytes(GGUF_MAGIC, 99, 0, 0);
        let result = GgufHeader::from_bytes(&buf);
        assert!(matches!(result, Err(GgufError::UnsupportedVersion(99))));
    }

    #[test]
    fn reject_truncated_header() {
        let buf = [0u8; 10];
        let result = GgufHeader::from_bytes(&buf);
        assert!(matches!(result, Err(GgufError::TruncatedFile { .. })));
    }
}
