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
 /// Duplicate the existing open file descriptor without reopening a path.
 /// On Unix all reads use positional I/O, so shared descriptor offsets
 /// cannot corrupt simultaneous expert reads.
 pub fn try_clone(&self) -> io::Result<Self> {
  Ok(Self { file: self.file.try_clone()?, path: self.path.clone(), file_size: self.file_size })
 }
 pub fn path(&self) -> &Path { &self.path }
 pub fn file_size(&self) -> u64 { self.file_size }
 /// Best-effort Linux page-cache eviction hint for read-only benchmark work.
 /// This never drops global caches, never requires root and never promises a cold read:
 /// other processes can keep or immediately repopulate the same pages.
 /// Non-Linux platforms report Unsupported rather than silently claiming success.
 pub fn advise_drop_cache(&self) -> io::Result<()> {
  #[cfg(target_os = "linux")]
  {
   use std::os::fd::AsRawFd;
   // POSIX_FADV_DONTNEED is 4 on Linux; 0 length applies to the entire file.
   unsafe extern "C" {
    fn posix_fadvise(fd: std::os::raw::c_int, offset: i64, len: i64, advice: std::os::raw::c_int) -> std::os::raw::c_int;
   }
   let errno = unsafe { posix_fadvise(self.file.as_raw_fd(), 0, 0, 4) };
   if errno == 0 { Ok(()) } else { Err(io::Error::from_raw_os_error(errno)) }
  }
  #[cfg(not(target_os = "linux"))]
  {
   Err(io::Error::new(io::ErrorKind::Unsupported, "page-cache eviction hint is only supported on Linux"))
  }
 }
 pub fn read_exact_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), RangeReadError> {
  let length = u64::try_from(buffer.len()).map_err(|_| RangeReadError::LengthTooLarge { length: u64::MAX })?;
  let end = offset.checked_add(length).ok_or(RangeReadError::OutOfBounds { offset, length, file_size: self.file_size })?;
  if end > self.file_size { return Err(RangeReadError::OutOfBounds { offset, length, file_size: self.file_size }); }
  #[cfg(unix)]
  {
   use std::os::unix::fs::FileExt;
   let mut done = 0usize;
   while done < buffer.len() {
    match self.file.read_at(&mut buffer[done..], offset + done as u64) {
     Ok(0) => return Err(RangeReadError::Io(io::Error::from(io::ErrorKind::UnexpectedEof))),
     Ok(count) => done += count,
     Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
     Err(error) => return Err(RangeReadError::Io(error)),
    }
   }
   Ok(())
  }
  #[cfg(not(unix))]
  {
   self.file.seek(SeekFrom::Start(offset))?;
   self.file.read_exact(buffer)?;
   Ok(())
  }
 }
 pub fn read_range(&mut self, offset: u64, length: u64) -> Result<Vec<u8>, RangeReadError> {
  // Validate against the file before allocating. Otherwise an invalid request
  // for a huge range could attempt a huge allocation before failing the read.
  let end = offset.checked_add(length).ok_or(RangeReadError::OutOfBounds {
   offset, length, file_size: self.file_size
  })?;
  if end > self.file_size {
   return Err(RangeReadError::OutOfBounds { offset, length, file_size: self.file_size });
  }
  let size = usize::try_from(length).map_err(|_| RangeReadError::LengthTooLarge { length })?;
  let mut bytes = vec![0; size];
  self.read_exact_at(offset, &mut bytes)?;
  Ok(bytes)
 }
}

#[cfg(test)]
mod tests {
 use super::*; use std::io::Write;
 fn temp_file() -> (PathBuf, Vec<u8>) { let path = std::env::temp_dir().join(format!("kiln-range-reader-{}-{}.bin", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())); let bytes: Vec<u8> = (0..=255).cycle().take(8192).collect(); File::create(&path).unwrap().write_all(&bytes).unwrap(); (path, bytes) }
 #[test] fn reads_requested_range() { let (path, expected) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert_eq!(reader.read_range(1024, 512).unwrap(), expected[1024..1536]); std::fs::remove_file(path).unwrap(); }
 #[test] fn rejects_out_of_bounds_and_overflow() { let (path, _) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert!(matches!(reader.read_range(8191, 2), Err(RangeReadError::OutOfBounds { .. }))); assert!(matches!(reader.read_range(u64::MAX, 2), Err(RangeReadError::OutOfBounds { .. }))); assert!(matches!(reader.read_range(0, u64::MAX), Err(RangeReadError::OutOfBounds { .. }))); std::fs::remove_file(path).unwrap(); }
 #[test] fn cache_advice_preserves_correct_read_semantics() {
  let (path, expected) = temp_file();
  let mut reader = FileRangeReader::open(&path).unwrap();
  assert_eq!(reader.read_range(0, 512).unwrap(), expected[..512]);
  #[cfg(target_os = "linux")]
  reader.advise_drop_cache().unwrap();
  #[cfg(not(target_os = "linux"))]
  assert_eq!(reader.advise_drop_cache().unwrap_err().kind(), io::ErrorKind::Unsupported);
  assert_eq!(reader.read_range(0, 512).unwrap(), expected[..512]);
  std::fs::remove_file(path).unwrap();
 }
 #[test] fn independent_positional_file_clones_do_not_share_seek_cursor() {
  let (path, data) = temp_file();
  let original = FileRangeReader::open(&path).unwrap();
  let mut threads = Vec::new();
  for id in 0..8usize {
   let mut fork = original.try_clone().unwrap();
   let expected = data[id * 1024..(id + 1) * 1024].to_vec();
   threads.push(std::thread::spawn(move || {
    for _ in 0..50 {
     assert_eq!(fork.read_range(id as u64 * 1024, 1024).unwrap(), expected);
    }
   }));
  }
  for thread in threads { thread.join().unwrap(); }
  std::fs::remove_file(path).unwrap();
 }
 #[test] fn allows_empty_range_at_eof() { let (path, expected) = temp_file(); let mut reader = FileRangeReader::open(&path).unwrap(); assert!(reader.read_range(expected.len() as u64, 0).unwrap().is_empty()); std::fs::remove_file(path).unwrap(); }
}
