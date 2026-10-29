//! Expert layout index for MoE models.
//!
//! The experts.bin file is expert-contiguous. This index maps expert IDs to
//! byte ranges. It is a flat list sorted by (layer, expert_id) so that the
//! whole-layer prefill path can load all experts for one layer with a single
//! sequential read pattern.

use serde::{Deserialize, Serialize};

/// A single expert byte range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRange {
    pub layer: u32,
    pub expert_id: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// The full expert table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertTable {
    /// Format version of the blob.
    pub blob_version: u32,

    /// Total bytes in the blob.
    pub blob_size: u64,

    /// All expert ranges, sorted by (layer, expert_id).
    pub ranges: Vec<ExpertRange>,
}

/// Errors found when validating an expert index against its blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertTableError {
    BlobSizeMismatch { indexed: u64, actual: u64 },
    EmptyRange { layer: u32, expert_id: u32 },
    RangeOutOfBounds { layer: u32, expert_id: u32, offset: u64, length: u64, blob_size: u64 },
    UnsortedEntries { previous_layer: u32, previous_expert: u32, layer: u32, expert_id: u32 },
    DuplicateExpert { layer: u32, expert_id: u32 },
    OverlappingRanges { first_offset: u64, first_end: u64, second_offset: u64 },
}

impl std::fmt::Display for ExpertTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlobSizeMismatch { indexed, actual } => {
                write!(f, "expert index declares {indexed} blob bytes but file has {actual}")
            }
            Self::EmptyRange { layer, expert_id } => {
                write!(f, "expert ({layer}, {expert_id}) has an empty byte range")
            }
            Self::RangeOutOfBounds { layer, expert_id, offset, length, blob_size } => write!(
                f,
                "expert ({layer}, {expert_id}) range at {offset} with length {length} exceeds blob size {blob_size}"
            ),
            Self::UnsortedEntries { previous_layer, previous_expert, layer, expert_id } => write!(
                f,
                "expert index is not sorted: ({previous_layer}, {previous_expert}) precedes ({layer}, {expert_id})"
            ),
            Self::DuplicateExpert { layer, expert_id } => {
                write!(f, "expert index contains duplicate entry ({layer}, {expert_id})")
            },
            Self::OverlappingRanges { first_offset, first_end, second_offset } => write!(
                f,
                "expert ranges overlap: first range ends at {first_end}, next starts at {second_offset} (first offset {first_offset})"
            ),
        }
    }
}

impl std::error::Error for ExpertTableError {}

impl ExpertTable {
    /// Validate index invariants before using any range to read the blob.
    /// Gaps are allowed; empty, overflowing, out-of-bounds, unsorted, or
    /// overlapping ranges are rejected.
    pub fn validate(&self, actual_blob_size: u64) -> Result<(), ExpertTableError> {
        if self.blob_size != actual_blob_size {
            return Err(ExpertTableError::BlobSizeMismatch {
                indexed: self.blob_size,
                actual: actual_blob_size,
            });
        }

        for pair in self.ranges.windows(2) {
            let previous = &pair[0];
            let current = &pair[1];
            if (current.layer, current.expert_id) == (previous.layer, previous.expert_id) {
                return Err(ExpertTableError::DuplicateExpert {
                    layer: current.layer,
                    expert_id: current.expert_id,
                });
            }
            if (current.layer, current.expert_id) < (previous.layer, previous.expert_id) {
                return Err(ExpertTableError::UnsortedEntries {
                    previous_layer: previous.layer,
                    previous_expert: previous.expert_id,
                    layer: current.layer,
                    expert_id: current.expert_id,
                });
            }
        }

        let mut by_offset: Vec<&ExpertRange> = self.ranges.iter().collect();
        for range in &self.ranges {
            if range.byte_length == 0 {
                return Err(ExpertTableError::EmptyRange {
                    layer: range.layer,
                    expert_id: range.expert_id,
                });
            }
            let end = range.byte_offset.checked_add(range.byte_length);
            if end.map_or(true, |end| end > self.blob_size) {
                return Err(ExpertTableError::RangeOutOfBounds {
                    layer: range.layer,
                    expert_id: range.expert_id,
                    offset: range.byte_offset,
                    length: range.byte_length,
                    blob_size: self.blob_size,
                });
            }
        }
        by_offset.sort_by_key(|range| range.byte_offset);
        for pair in by_offset.windows(2) {
            let first = pair[0];
            let second = pair[1];
            let first_end = first.byte_offset + first.byte_length;
            if second.byte_offset < first_end {
                return Err(ExpertTableError::OverlappingRanges {
                    first_offset: first.byte_offset,
                    first_end,
                    second_offset: second.byte_offset,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(layer: u32, expert_id: u32, byte_offset: u64, byte_length: u64) -> ExpertRange {
        ExpertRange { layer, expert_id, byte_offset, byte_length }
    }

    fn table(ranges: Vec<ExpertRange>, blob_size: u64) -> ExpertTable {
        ExpertTable { blob_version: 1, blob_size, ranges }
    }

    #[test]
    fn accepts_sorted_non_overlapping_ranges_with_gaps() {
        let index = table(vec![range(0, 0, 0, 8), range(0, 1, 16, 8), range(1, 0, 8, 8)], 24);
        assert!(index.validate(24).is_ok());
    }

    #[test]
    fn rejects_blob_size_mismatch() {
        assert!(matches!(
            table(vec![], 10).validate(11),
            Err(ExpertTableError::BlobSizeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_unsorted_entries() {
        let index = table(vec![range(1, 0, 0, 8), range(0, 0, 8, 8)], 16);
        assert!(matches!(index.validate(16), Err(ExpertTableError::UnsortedEntries { .. })));
    }

    #[test]
    fn rejects_empty_and_out_of_bounds_ranges() {
        assert!(matches!(
            table(vec![range(0, 0, 0, 0)], 16).validate(16),
            Err(ExpertTableError::EmptyRange { .. })
        ));
        assert!(matches!(
            table(vec![range(0, 0, u64::MAX, 2)], u64::MAX).validate(u64::MAX),
            Err(ExpertTableError::RangeOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_expert_entries() {
        let index = table(vec![range(0, 0, 0, 8), range(0, 0, 8, 8)], 16);
        assert!(matches!(index.validate(16), Err(ExpertTableError::DuplicateExpert { .. })));
    }

    #[test]
    fn rejects_overlapping_byte_ranges() {
        let index = table(vec![range(0, 0, 0, 10), range(0, 1, 8, 8)], 32);
        assert!(matches!(index.validate(32), Err(ExpertTableError::OverlappingRanges { .. })));
    }
}
