//! Phase 5.3 Linux expert-range reader using io_uring with safe CPU fallback.
//!
//! Portable, buffered reads remain the default fallback. SQPOLL is strictly
//! opt-in: the kernel may deny setup or require features not present on Tier 0.
//! Each ring request uses a dedicated, fixed-size buffer kept alive until its
//! completion CQE is reaped. If submission fails ambiguously, that single
//! 64 KiB allocation is deliberately leaked and the ring is disabled; the
//! reader subsequently uses pread. This sacrifices <=64 KiB on a fatal driver
//! error rather than risking a use-after-free from a kernel-owned raw pointer.
use std::fs::File;
use std::io::{self, ErrorKind};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use io_uring::{opcode, types, IoUring};
use kiln_models::ExpertTable;

const MAX_READ_CHUNK: usize = 64 * 1024;
const RING_DEPTH: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxExpertIoMode {
    BufferedFallback,
    IoUring,
    IoUringSqPoll,
}

pub struct LinuxUringExpertReader {
    source: File,
    table: ExpertTable,
    ring: Option<IoUring>,
    mode: LinuxExpertIoMode,
}

impl LinuxUringExpertReader {
    /// Open and validate a complete expert index before any file reads.
    /// SQPOLL failure automatically falls back to normal io_uring, and ring
    /// setup failure falls back to synchronous buffered pread on all kernels.
    pub fn open(
        path: &Path, table: ExpertTable, request_sqpoll: bool,
    ) -> io::Result<Self> {
        let source = File::open(path)?;
        let source_size = source.metadata()?.len();
        table.validate(source_size)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
        let (ring, mode) = if request_sqpoll {
            match IoUring::builder().setup_sqpoll(1000).build(RING_DEPTH) {
                Ok(ring) => (Some(ring), LinuxExpertIoMode::IoUringSqPoll),
                Err(_) => match IoUring::new(RING_DEPTH) {
                    Ok(ring) => (Some(ring), LinuxExpertIoMode::IoUring),
                    Err(_) => (None, LinuxExpertIoMode::BufferedFallback),
                },
            }
        } else {
            match IoUring::new(RING_DEPTH) {
                Ok(ring) => (Some(ring), LinuxExpertIoMode::IoUring),
                Err(_) => (None, LinuxExpertIoMode::BufferedFallback),
            }
        };
        Ok(Self { source, table, ring, mode })
    }

    pub fn mode(&self) -> LinuxExpertIoMode { self.mode }

    fn expert_range(&self, layer: u32, expert_id: u32) -> io::Result<(u64, u64)> {
        let idx = self.table.ranges.binary_search_by_key(
            &(layer, expert_id), |e| (e.layer, e.expert_id),
        ).map_err(|_| io::Error::new(
            ErrorKind::NotFound, format!("expert ({layer}, {expert_id}) not indexed"),
        ))?;
        let info = &self.table.ranges[idx];
        Ok((info.byte_offset, info.byte_length))
    }

    /// Exactly read one validated expert without copying any unrelated model
    /// bytes. This method returns owned output; callers may then insert it
    /// into the byte-bounded shared expert cache.
    pub fn read_expert(&mut self, layer: u32, expert_id: u32) -> io::Result<Vec<u8>> {
        let (start, len) = self.expert_range(layer, expert_id)?;
        let count = usize::try_from(len).map_err(|_| {
            io::Error::new(ErrorKind::InvalidInput, "expert exceeds address space")
        })?;
        let mut output = vec![0u8; count];
        self.read_exact_at(start, &mut output)?;
        Ok(output)
    }

    pub fn read_expert_into(
        &mut self, layer: u32, expert_id: u32, dst: &mut [u8],
    ) -> io::Result<()> {
        let (start, len) = self.expert_range(layer, expert_id)?;
        if dst.len() as u64 != len {
            return Err(io::Error::new(ErrorKind::InvalidInput,
                format!("buffer size {}, expected {len}", dst.len())));
        }
        self.read_exact_at(start, dst)
    }

    /// Read in bounded segments so native CQE lengths are always <=64 KiB.
    fn read_exact_at(&mut self, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
        while !dst.is_empty() {
            let amount = dst.len().min(MAX_READ_CHUNK);
            let (segment, rest) = dst.split_at_mut(amount);
            if self.ring.is_some() {
                let mut completed = 0usize;
                while completed < segment.len() {
                    let got = self.read_ring_once(offset + completed as u64,
                        &mut segment[completed..])?;
                    if got == 0 {
                        return Err(io::Error::new(ErrorKind::UnexpectedEof, "empty io_uring expert read"));
                    }
                    completed += got;
                }
            } else {
                self.source.read_exact_at(segment, offset)?;
            }
            offset = offset.checked_add(amount as u64).ok_or_else(|| {
                io::Error::new(ErrorKind::InvalidInput, "read offset overflow")
            })?;
            dst = rest;
        }
        Ok(())
    }

    /// Read a single <=64 KiB request. A boxed private buffer is submitted
    /// to the kernel; the caller's destination is never handed to io_uring.
    fn read_ring_once(&mut self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        let Some(ring) = self.ring.as_mut() else {
            self.source.read_exact_at(dst, offset)?;
            return Ok(dst.len());
        };
        let mut kernel_buffer = Box::new([0u8; MAX_READ_CHUNK]);
        let requested = dst.len().min(MAX_READ_CHUNK);
        let entry = opcode::Read::new(
            types::Fd(self.source.as_raw_fd()),
            kernel_buffer.as_mut_ptr(), requested as u32,
        ).offset(offset).build().user_data(1);

        // No kernel ownership is transferred if pushing the SQE fails.
        if unsafe { ring.submission().push(&entry) }.is_err() {
            self.mode = LinuxExpertIoMode::BufferedFallback;
            self.ring = None;
            self.source.read_exact_at(dst, offset)?;
            return Ok(dst.len());
        }

        // Once queued, the kernel may own the raw pointer until completion.
        // If entering the ring fails ambiguously, leak exactly this private
        // buffer, drop/disable the ring, and perform safe buffered fallback.
        // It is never safe to free the box before the CQE is reaped.
        if ring.submit_and_wait(1).is_err() {
            std::mem::forget(kernel_buffer);
            self.mode = LinuxExpertIoMode::BufferedFallback;
            self.ring = None;
            self.source.read_exact_at(dst, offset)?;
            return Ok(dst.len());
        }
        let completed = ring.completion().next();
        let Some(cqe) = completed else {
            std::mem::forget(kernel_buffer);
            self.mode = LinuxExpertIoMode::BufferedFallback;
            self.ring = None;
            self.source.read_exact_at(dst, offset)?;
            return Ok(dst.len());
        };
        let raw = cqe.result();
        if raw < 0 {
            // A completion reaps the kernel's buffer access; this error can
            // safely use synchronous fallback without retaining the box.
            self.mode = LinuxExpertIoMode::BufferedFallback;
            self.ring = None;
            self.source.read_exact_at(dst, offset)?;
            return Ok(dst.len());
        }
        let count = usize::try_from(raw).map_err(|_| {
            io::Error::new(ErrorKind::InvalidData, "negative ring completion")
        })?;
        if count > requested {
            return Err(io::Error::new(ErrorKind::InvalidData, "io_uring read exceeded buffer"));
        }
        dst[..count].copy_from_slice(&kernel_buffer[..count]);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_models::ExpertRange;

    #[test]
    fn exact_expert_reads_and_fallback_are_byte_identical() {
        let root=std::env::temp_dir().join(format!(
            "kiln-uring-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("experts.bin");
        let bytes:Vec<u8>=(0..=255u8).cycle().take(140_000).collect();
        std::fs::write(&path,&bytes).unwrap();
        let table=ExpertTable {blob_version:1,blob_size:bytes.len() as u64,
            ranges:vec![
                ExpertRange{layer:0,expert_id:0,byte_offset:0,byte_length:70_000},
                ExpertRange{layer:0,expert_id:1,byte_offset:70_000,byte_length:65_000},
            ],
        };
        let mut reader=LinuxUringExpertReader::open(&path,table,true).unwrap();
        assert_eq!(reader.read_expert(0,0).unwrap(),bytes[..70_000]);
        let mut buffer=vec![0;65_000];
        reader.read_expert_into(0,1,&mut buffer).unwrap();
        assert_eq!(buffer,bytes[70_000..135_000]);
        assert!(reader.read_expert(0,99).is_err());
        assert!(reader.read_expert_into(0,1,&mut [0u8;5]).is_err());
        drop(reader);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn invalid_expert_ranges_fail_before_ring_setup() {
        let root=std::env::temp_dir().join(format!(
            "kiln-uring-bad-{}-{:?}",std::process::id(),std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path=root.join("source");
        std::fs::write(&path,[1u8;16]).unwrap();
        let table=ExpertTable{blob_version:1,blob_size:16,ranges:vec![
            ExpertRange{layer:0,expert_id:0,byte_offset:12,byte_length:8},
        ]};
        assert!(LinuxUringExpertReader::open(&path,table,false).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
