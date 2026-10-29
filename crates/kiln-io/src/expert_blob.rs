//! Validated expert-index access over a large expert blob.

use std::path::Path;

use kiln_models::{ExpertTable, ExpertTableError};

use crate::{FileRangeReader, RangeReadError};

#[derive(Debug)]
pub enum ExpertBlobError {
    Io(RangeReadError),
    InvalidIndex(ExpertTableError),
    MissingExpert { layer: u32, expert_id: u32 },
    BufferLengthMismatch { expected: u64, actual: usize },
}

impl std::fmt::Display for ExpertBlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
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
