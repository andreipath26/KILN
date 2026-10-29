//! Bounds-checked random access to large model/expert blobs.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum RangeReadError { Io(io::Error), OutOfBounds { offset: u64, length: u64, file_size: u64 }, LengthTooLarge { length: u64 } }
impl std::fmt::Display for RangeReadError {
 fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self { Self::Io(e) => write!(f, "range read failed: {e}"), Self::OutOfBounds { offset, length, file_size } => write!(f, "range at {offset} with length {length} exceeds file size {file_size}"), Self::LengthTooLarge { length } => write!(f, "range length {length} cannot fit in memory") } }
}
impl std::error::Error for RangeReadError {}
impl From<io::Error> for RangeReadError { fn from(e: io::Error) -> Self { Self::Io(e) } }

/// Reads only requested ranges; never maps or loads the entire blob.
pub struct FileRangeReader { file: File, path: PathBuf, file_size: u64 }
impl FileRangeReader {
 pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeReadError> { let path = path.as_ref().to_path_buf(); let file = File::open(&path)?; let file_size = file.metadata()?.len(); Ok(Self { file, path, file_size }) }
 pub fn path(&self) -> &Path { &self.path }
 pub fn file_size(&self) -> u64 { self.file_size }
 pub fn read_exact_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), RangeReadError> {
  let length = u64::try_from(buffer.len()).map_err(|_| RangeReadError::LengthTooLarge { length: u64::MAX })?;
  let end = offset.checked_add(length).ok_or(RangeReadError::OutOfBounds { offset, length, file_size: self.file_size })?;
  if end > self.file_size { return Err(RangeReadError::OutOfBounds { offset, length, file_size: self.file_size }); }
  self.file.seek(SeekFrom::Start(offset))?; self.file.read_exact(buffer)?; Ok(())
 }
 pub fn read_range(&mut self, offset: u64, length: u64) -> Result<Vec<u8>, RangeReadError> { let size = usize::try_from(length).map_err(|_| RangeReadError::LengthTooLarge { length })?; let mut bytes = vec![0; size]; self.read_exact_at(offset, &mut bytes)?; Ok(bytes) }
}

#[cfg(test)]
mod tests {
 use super::*; use std::io::Write;
 fn temp_file() -> (PathBuf, Vec<u8>) { let path = std::env::temp_dir().join(format!("kiln-range-reader-{}-{}.bin", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())); let bytes: Vec<u8> = (0..=255).cycle().take(8192).collect(); File::create(&path).unwrap().write_all(&bytes).unwrap(); (path, bytes) }
 #[test] fn reads_requested_range() { let (path, expected) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert_eq!(reader.read_range(1024, 512).unwrap(), expected[1024..1536]); std::fs::remove_file(path).unwrap(); }
 #[test] fn rejects_out_of_bounds_and_overflow() { let (path, _) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert!(matches!(reader.read_range(8191, 2), Err(RangeReadError::OutOfBounds { .. }))); assert!(matches!(reader.read_range(u64::MAX, 2), Err(RangeReadError::OutOfBounds { .. }))); std::fs::remove_file(path).unwrap(); }
 #[test] fn allows_empty_range_at_eof() { let (path, expected) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert!(reader.read_range(expected.len() as u64, 0).unwrap().is_empty()); std::fs::remove_file(path).unwrap(); }
}
