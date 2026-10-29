//! Validated expert-index access over a large expert blob.

use std::path::Path;

use kiln_models::{ExpertTable, ExpertTableError};

use crate::{FileRangeReader, RangeReadError};

const MAX_INDEX_FILE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug)]
pub enum ExpertBlobError {
    Io(RangeReadError),
    IndexFileIo(std::io::Error),
    InvalidIndexJson(serde_json::Error),
    IndexFileTooLarge { actual_bytes: u64, max_bytes: u64 },
    InvalidIndex(ExpertTableError),
    MissingExpert { layer: u32, expert_id: u32 },
    BufferLengthMismatch { expected: u64, actual: usize },
}

impl std::fmt::Display for ExpertBlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::IndexFileIo(error) => write!(f, "expert index file I/O error: {error}"),
            Self::InvalidIndexJson(error) => write!(f, "invalid expert index JSON: {error}"),
            Self::IndexFileTooLarge { actual_bytes, max_bytes } => write!(
                f, "expert index file is {actual_bytes} bytes; limit is {max_bytes} bytes"
            ),
            Self::InvalidIndex(error) => write!(f, "invalid expert index: {error}"),
            Self::MissingExpert { layer, expert_id } => {
                write!(f, "expert ({layer}, {expert_id}) is not present in the index")
            }
            Self::BufferLengthMismatch { expected, actual } => {
                write!(f, "expert buffer has {actual} bytes; expected exactly {expected}")
            }
        }
    }
}

impl std::error::Error for ExpertBlobError {}

impl From<RangeReadError> for ExpertBlobError {
    fn from(value: RangeReadError) -> Self {
        Self::Io(value)
    }
}

impl From<ExpertTableError> for ExpertBlobError {
    fn from(value: ExpertTableError) -> Self {
        Self::InvalidIndex(value)
    }
}

/// Opens a large expert blob only after validating every indexed range.
/// It keeps the file handle open and reads only the requested expert bytes.
pub struct ExpertBlobReader {
    reader: FileRangeReader,
    table: ExpertTable,
}

impl ExpertBlobReader {
    pub fn open(
        path: impl AsRef<Path>,
        table: ExpertTable,
    ) -> Result<Self, ExpertBlobError> {
        let reader = FileRangeReader::open(path)?;
        table.validate(reader.file_size())?;
        Ok(Self { reader, table })
    }

    /// Load and validate a JSON ExpertTable before opening the indexed blob.
    /// The index file is size-limited to avoid unbounded metadata allocation.
    pub fn open_indexed(
        path: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
    ) -> Result<Self, ExpertBlobError> {
        let index_path = index_path.as_ref();
        let index_size = std::fs::metadata(index_path).map_err(ExpertBlobError::IndexFileIo)?.len();
        if index_size > MAX_INDEX_FILE_BYTES {
            return Err(ExpertBlobError::IndexFileTooLarge {
                actual_bytes: index_size,
                max_bytes: MAX_INDEX_FILE_BYTES,
            });
        }
        let bytes = std::fs::read(index_path).map_err(ExpertBlobError::IndexFileIo)?;
        let table: ExpertTable = serde_json::from_slice(&bytes).map_err(ExpertBlobError::InvalidIndexJson)?;
        Self::open(path, table)
    }

    pub fn file_size(&self) -> u64 {
        self.reader.file_size()
    }

    pub fn table(&self) -> &ExpertTable {
        &self.table
    }

    pub fn read_expert(
        &mut self,
        layer: u32,
        expert_id: u32,
    ) -> Result<Vec<u8>, ExpertBlobError> {
        let index = self.table.ranges.binary_search_by_key(
            &(layer, expert_id),
            |range| (range.layer, range.expert_id),
        ).map_err(|_| ExpertBlobError::MissingExpert { layer, expert_id })?;
        let range = &self.table.ranges[index];
        Ok(self.reader.read_range(range.byte_offset, range.byte_length)?)
    }

    /// Read into a caller-owned buffer to allow reuse across token steps.
    pub fn read_expert_into(
        &mut self,
        layer: u32,
        expert_id: u32,
        buffer: &mut [u8],
    ) -> Result<(), ExpertBlobError> {
        let index = self.table.ranges.binary_search_by_key(
            &(layer, expert_id),
            |range| (range.layer, range.expert_id),
        ).map_err(|_| ExpertBlobError::MissingExpert { layer, expert_id })?;
        let range = &self.table.ranges[index];
        if buffer.len() as u64 != range.byte_length {
            return Err(ExpertBlobError::BufferLengthMismatch {
                expected: range.byte_length,
                actual: buffer.len(),
            });
        }
        self.reader.read_exact_at(range.byte_offset, buffer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_models::ExpertRange;
    use std::io::Write;

    fn fixture() -> (std::path::PathBuf, Vec<u8>) {
        let path = std::env::temp_dir().join(format!(
            "kiln-expert-blob-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes: Vec<u8> = (0..32).collect();
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        (path, bytes)
    }

    fn index(size: u64) -> ExpertTable {
        ExpertTable {
            blob_version: 1,
            blob_size: size,
            ranges: vec![
                ExpertRange { layer: 0, expert_id: 0, byte_offset: 0, byte_length: 8 },
                ExpertRange { layer: 0, expert_id: 1, byte_offset: 8, byte_length: 8 },
                ExpertRange { layer: 1, expert_id: 0, byte_offset: 16, byte_length: 8 },
            ],
        }
    }

    #[test]
    fn reads_only_the_requested_expert() {
        let (path, bytes) = fixture();
        let mut reader = ExpertBlobReader::open(&path, index(bytes.len() as u64)).unwrap();
        assert_eq!(reader.read_expert(0, 1).unwrap(), bytes[8..16]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn validates_index_before_any_expert_read() {
        let (path, bytes) = fixture();
        let mut bad = index(bytes.len() as u64);
        bad.ranges[1].byte_offset = 4;
        assert!(matches!(
            ExpertBlobReader::open(&path, bad),
            Err(ExpertBlobError::InvalidIndex(ExpertTableError::OverlappingRanges { .. }))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn supports_reusable_caller_buffer_and_rejects_wrong_size() {
        let (path, bytes) = fixture();
        let mut reader = ExpertBlobReader::open(&path, index(bytes.len() as u64)).unwrap();
        let mut buffer = [0u8; 8];
        reader.read_expert_into(1, 0, &mut buffer).unwrap();
        assert_eq!(buffer.as_slice(), &bytes[16..24]);
        assert!(matches!(
            reader.read_expert_into(1, 0, &mut [0u8; 4]),
            Err(ExpertBlobError::BufferLengthMismatch { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn opens_validated_json_index_and_rejects_invalid_json() {
        let (path, bytes) = fixture();
        let index_path = std::env::temp_dir().join(format!(
            "kiln-expert-blob-index-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::write(&index_path, serde_json::to_vec(&index(bytes.len() as u64)).unwrap()).unwrap();
        let mut reader = ExpertBlobReader::open_indexed(&path, &index_path).unwrap();
        assert_eq!(reader.read_expert(1, 0).unwrap(), bytes[16..24]);
        std::fs::write(&index_path, "{invalid").unwrap();
        assert!(matches!(
            ExpertBlobReader::open_indexed(&path, &index_path),
            Err(ExpertBlobError::InvalidIndexJson(_))
        ));
        std::fs::remove_file(index_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_missing_expert() {
        let (path, bytes) = fixture();
        let mut reader = ExpertBlobReader::open(&path, index(bytes.len() as u64)).unwrap();
        assert!(matches!(
            reader.read_expert(9, 9),
            Err(ExpertBlobError::MissingExpert { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }
}
