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

/// The nine GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            9 => Ok(GgufValueType::Array),
            other => Err(GgufError::Metadata(format!("unknown value type: {}", other))),
        }
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufValueType,
        values: Vec<GgufValue>,
    },
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int8(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Int16(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn need(&self, n: usize) -> Result<(), GgufError> {
        if self.remaining() < n {
            return Err(GgufError::TruncatedFile {
                needed: n,
                got: self.remaining(),
            });
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        self.need(4)?;
        let v = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.u32()? as i32)
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        self.need(8)?;
        let v = u64::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
            self.bytes[self.pos + 4],
            self.bytes[self.pos + 5],
            self.bytes[self.pos + 6],
            self.bytes[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.u8()? != 0)
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + len])
            .map_err(|e| GgufError::Metadata(format!("invalid UTF-8: {}", e)))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn value(&mut self, t: GgufValueType) -> Result<GgufValue, GgufError> {
        match t {
            GgufValueType::Uint8 => Ok(GgufValue::Uint8(self.u8()?)),
            GgufValueType::Int8 => Ok(GgufValue::Int8(self.i8()?)),
            GgufValueType::Uint16 => Ok(GgufValue::Uint16(self.u16()?)),
            GgufValueType::Int16 => Ok(GgufValue::Int16(self.i16()?)),
            GgufValueType::Uint32 => Ok(GgufValue::Uint32(self.u32()?)),
            GgufValueType::Int32 => Ok(GgufValue::Int32(self.i32()?)),
            GgufValueType::Float32 => Ok(GgufValue::Float32(self.f32()?)),
            GgufValueType::Bool => Ok(GgufValue::Bool(self.bool()?)),
            GgufValueType::String => Ok(GgufValue::String(self.string()?)),
            GgufValueType::Array => {
                let element_type_raw = self.u32()?;
                let element_type = GgufValueType::from_u32(element_type_raw)?;
                let count = self.u64()? as usize;
                let mut values = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    values.push(self.value(element_type)?);
                }
                Ok(GgufValue::Array { element_type, values })
            }
        }
    }
}

/// Parse the metadata key-value table from a byte slice starting at the
/// offset immediately after the header.
pub fn parse_metadata(
    bytes: &[u8],
    start_offset: usize,
    kv_count: u64,
) -> Result<(std::collections::HashMap<String, GgufValue>, usize), GgufError> {
    let mut reader = Reader::new(&bytes[start_offset..]);
    let mut map = std::collections::HashMap::new();

    for i in 0..kv_count {
        let key = reader.string().map_err(|e| {
            GgufError::Metadata(format!("key {} read failed: {}", i, e))
        })?;
        let type_raw = reader.u32().map_err(|e| {
            GgufError::Metadata(format!("value type for key '{}' failed: {}", key, e))
        })?;
        let vt = GgufValueType::from_u32(type_raw).map_err(|e| {
            GgufError::Metadata(format!("value type for key '{}' failed: {}", key, e))
        })?;
        let value = reader.value(vt).map_err(|e| {
            GgufError::Metadata(format!("value for key '{}' failed: {}", key, e))
        })?;
        map.insert(key, value);
    }

    Ok((map, start_offset + reader.pos))
}

/// The dtype of a GGUF tensor. KILN supports three, rejects the rest.
///
/// No explicit discriminants because Unknown(u32) carries data.
/// The numeric codes are mapped in from_u32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    F32,
    F16,
    Tq1_0,
    Unknown(u32),
}

impl GgufType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgufType::F32,
            1 => GgufType::F16,
            1000 => GgufType::Tq1_0,
            other => GgufType::Unknown(other),
        }
    }

    /// The byte size of one element of this type.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            GgufType::F32 => Some(4),
            GgufType::F16 => Some(2),
            // TQ1_0 packs 5 trits into 1 byte. Element size is fractional,
            // so we return None and callers compute the byte size from the
            // shape with the packed formula.
            GgufType::Tq1_0 => None,
            GgufType::Unknown(_) => None,
        }
    }
}

/// One tensor in the GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// The number of elements in the tensor.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The byte size of the tensor.
    pub fn byte_size(&self) -> Option<u64> {
        let n = self.num_elements();
        match self.dtype {
            GgufType::F32 => Some(n * 4),
            GgufType::F16 => Some(n * 2),
            GgufType::Tq1_0 => Some((n + 4) / 5),
            GgufType::Unknown(_) => None,
        }
    }
}

/// Parse the tensor table from a byte slice starting at the offset
/// immediately after the metadata table.
///
/// Returns the tensor list and the byte offset where the tensor table
/// ends and the data section begins.
pub fn parse_tensor_table(
    bytes: &[u8],
    start_offset: usize,
    tensor_count: u64,
) -> Result<(Vec<GgufTensorInfo>, usize), GgufError> {
    let mut reader = Reader::new(&bytes[start_offset..]);
    let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);

    for i in 0..tensor_count {
        let name = reader.string().map_err(|e| {
            GgufError::Tensor(format!("tensor {} name read failed: {}", i, e))
        })?;
        let dims = reader.u32().map_err(|e| {
            GgufError::Tensor(format!("tensor '{}' ndims read failed: {}", name, e))
        })?;
        if dims > 8 {
            return Err(GgufError::Tensor(format!(
                "tensor '{}' has {} dimensions, max is 8",
                name, dims
            )));
        }
        let mut shape = Vec::with_capacity(dims as usize);
        for d in 0..dims {
            let v = reader.u64().map_err(|e| {
                GgufError::Tensor(format!(
                    "tensor '{}' dim {} read failed: {}",
                    name, d, e
                ))
            })?;
            shape.push(v);
        }
        let dtype_raw = reader.u32().map_err(|e| {
            GgufError::Tensor(format!("tensor '{}' dtype read failed: {}", name, e))
        })?;
        let dtype = GgufType::from_u32(dtype_raw);
        let offset = reader.u64().map_err(|e| {
            GgufError::Tensor(format!("tensor '{}' offset read failed: {}", name, e))
        })?;
        tensors.push(GgufTensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    Ok((tensors, start_offset + reader.pos))
}

/// A fully-loaded GGUF file with its data section mapped.
pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: std::collections::HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    mmap: memmap2::Mmap,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("header", &self.header)
            .field("metadata_count", &self.metadata.len())
            .field("tensor_count", &self.tensors.len())
            .field("data_offset", &self.data_offset)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl GgufFile {
    /// Load a GGUF file from disk.
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let bytes: &[u8] = &mmap;

        // Parse header.
        let header = GgufHeader::from_bytes(bytes)?;

        // Parse metadata.
        let (metadata, after_metadata) =
            parse_metadata(bytes, GgufHeader::SIZE, header.metadata_kv_count)?;

        // Parse tensor table.
        let (tensors, after_tensor_table) =
            parse_tensor_table(bytes, after_metadata, header.tensor_count)?;

        // Read the alignment from metadata. Default 32 if not present.
        let alignment: u64 = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(32);

        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(GgufError::Alignment(format!(
                "alignment {} is not a power of two",
                alignment
            )));
        }

        // Round the offset up to the alignment boundary.
        let data_offset = align_up(after_tensor_table as u64, alignment);

        // Verify that every tensor with a known dtype fits within the
        // file. Tensors with unknown dtypes are skipped. The loader
        // opens any GGUF file regardless of dtype. It only refuses to
        // decode tensors whose type it does not support, and that
        // refusal happens in tensor_bytes when the caller asks for
        // the bytes.
        for t in &tensors {
            let byte_size = match t.byte_size() {
                Some(n) => n,
                None => continue, // unknown dtype, skip bounds check
            };
            let end = data_offset
                .checked_add(t.offset)
                .and_then(|v| v.checked_add(byte_size))
                .ok_or_else(|| GgufError::Tensor(format!(
                    "tensor '{}' offset overflow",
                    t.name
                )))?;
            if end > bytes.len() as u64 {
                return Err(GgufError::Tensor(format!(
                    "tensor '{}' extends past end of file (end={}, file={})",
                    t.name,
                    end,
                    bytes.len()
                )));
            }
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    /// Look up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the raw bytes of a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensor(name)?;
        let byte_size = t.byte_size()? as usize;
        let start = (self.data_offset + t.offset) as usize;
        let end = start + byte_size;
        self.mmap.get(start..end)
    }

    /// The total bytes of the file.
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// The number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The number of metadata entries.
    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }

    /// Get a metadata string by key.
    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get a metadata u32 by key.
    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }
}

/// Round `value` up to the nearest multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
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

    #[test]
    fn parse_metadata_single_string() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"name");
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(b"abc");
        let (map, end) = parse_metadata(&buf, 0, 1).unwrap();
        assert_eq!(end, buf.len());
        assert_eq!(map.get("name").unwrap().as_str().unwrap(), "abc");
    }

    #[test]
    fn parse_metadata_mixed_types() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"a");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&42u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"b");
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&3.5f32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"c");
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(1);
        let (map, _) = parse_metadata(&buf, 0, 3).unwrap();
        assert_eq!(map.get("a").unwrap().as_u32().unwrap(), 42);
        assert_eq!(map.get("b").unwrap().as_f32().unwrap(), 3.5);
        assert!(map.get("c").unwrap().as_bool().unwrap());
    }

    #[test]
    fn parse_metadata_array_of_strings() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"tags");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(b"one");
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(b"two");
        let (map, _) = parse_metadata(&buf, 0, 1).unwrap();
        match map.get("tags").unwrap() {
            GgufValue::Array { element_type, values } => {
                assert_eq!(*element_type, GgufValueType::String);
                assert_eq!(values.len(), 2);
                assert_eq!(values[0].as_str().unwrap(), "one");
                assert_eq!(values[1].as_str().unwrap(), "two");
            }
            other => panic!("expected array, got {:?}", other),
        }
    }

    #[test]
    fn parse_metadata_rejects_unknown_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"x");
        buf.extend_from_slice(&99u32.to_le_bytes());
        let result = parse_metadata(&buf, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn parse_tensor_table_single_f32() {
        // One tensor: name "w", 2 dims [4, 3], dtype F32 (0), offset 0
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"w");
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndims
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let (tensors, end) = parse_tensor_table(&buf, 0, 1).unwrap();
        assert_eq!(end, buf.len());
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(t.name, "w");
        assert_eq!(t.shape, vec![4, 3]);
        assert_eq!(t.dtype, GgufType::F32);
        assert_eq!(t.offset, 0);
        assert_eq!(t.num_elements(), 12);
        assert_eq!(t.byte_size(), Some(48));
    }

    #[test]
    fn parse_tensor_table_tq1_0() {
        // A TQ1_0 tensor with 10 elements -> (10 + 4) / 5 = 2 bytes
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"tern");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&10u64.to_le_bytes());
        buf.extend_from_slice(&1000u32.to_le_bytes()); // Tq1_0
        buf.extend_from_slice(&0u64.to_le_bytes());

        let (tensors, _) = parse_tensor_table(&buf, 0, 1).unwrap();
        let t = &tensors[0];
        assert_eq!(t.dtype, GgufType::Tq1_0);
        assert_eq!(t.num_elements(), 10);
        assert_eq!(t.byte_size(), Some(2));
    }

    #[test]
    fn parse_tensor_table_unknown_dtype_preserved() {
        // An unknown dtype should be preserved as Unknown(n), not rejected.
        // The loader decides what to do with it later.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"x");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&9999u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let (tensors, _) = parse_tensor_table(&buf, 0, 1).unwrap();
        assert_eq!(tensors[0].dtype, GgufType::Unknown(9999));
        assert_eq!(tensors[0].byte_size(), None);
    }

    #[test]
    fn parse_tensor_table_rejects_too_many_dims() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"x");
        buf.extend_from_slice(&16u32.to_le_bytes()); // 16 dims, max is 8
        let result = parse_tensor_table(&buf, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn align_up_basics() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(0, 1), 0);
    }

    /// Build a minimal valid GGUF file in memory and write it to a
    /// temp path. Returns the path.
    fn write_minimal_gguf(name: &str) -> std::path::PathBuf {
        let mut buf = Vec::new();
        // Header: magic, version 3, tensor_count=1, metadata_count=1
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata: key "general.alignment", type Uint32, value 32
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // Uint32
        buf.extend_from_slice(&32u32.to_le_bytes());
        // Tensor table: 1 tensor "w", 1 dim [10], TQ1_0, offset 0
        let tensor_name = b"w";
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&10u64.to_le_bytes());
        buf.extend_from_slice(&1000u32.to_le_bytes()); // Tq1_0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Pad to alignment 32
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // Tensor data: 2 bytes for 10 trits
        buf.extend_from_slice(&[0x11, 0x22]);

        let mut p = std::env::temp_dir();
        p.push(format!("kiln_gguf_test_{}_{}.gguf", name, std::process::id()));
        std::fs::write(&p, &buf).unwrap();
        p
    }

    #[test]
    fn gguf_file_open_and_read_tensor() {
        let path = write_minimal_gguf("open");
        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.tensor_count(), 1);
        assert_eq!(g.metadata_count(), 1);
        assert_eq!(g.metadata_u32("general.alignment"), Some(32));
        let t = g.tensor("w").unwrap();
        assert_eq!(t.dtype, GgufType::Tq1_0);
        assert_eq!(t.shape, vec![10]);
        let bytes = g.tensor_bytes("w").unwrap();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0x11);
        assert_eq!(bytes[1], 0x22);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn gguf_file_missing_tensor_returns_none() {
        let path = write_minimal_gguf("missing");
        let g = GgufFile::open(&path).unwrap();
        assert!(g.tensor("does-not-exist").is_none());
        assert!(g.tensor_bytes("does-not-exist").is_none());
        std::fs::remove_file(&path).ok();
    }
}
