//! Verified Phase 5.1 component access over the existing shared expert cache.
//!
//! The packed sidecar's CRC32, expert ranges, and full tensor-component
//! coverage are verified once on open. Runtime reads return Arc-backed
//! read-only views, without cloning an expert for every tensor component.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use kiln_mem::{ExpertCachePolicy, ExpertCacheStats};

use crate::cached_expert_blob::CachedExpertBlobReader;
use crate::expert_sidecar::{verify_component_expert_sidecar, SidecarPackError};
use crate::expert_blob::ExpertBlobError;

#[derive(Debug)]
pub enum ComponentReadError {
    Pack(SidecarPackError),
    Blob(ExpertBlobError),
    NotFound { layer: u32, expert_id: u32, tensor_name: String },
    Invalid(String),
}
impl std::fmt::Display for ComponentReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pack(err) => write!(f, "{err}"),
            Self::Blob(err) => write!(f, "{err}"),
            Self::NotFound { layer, expert_id, tensor_name } =>
                write!(f, "missing component ({layer}, {expert_id}, {tensor_name})"),
            Self::Invalid(err) => write!(f, "component layout invalid: {err}"),
        }
    }
}
impl std::error::Error for ComponentReadError {}
impl From<SidecarPackError> for ComponentReadError {
    fn from(v: SidecarPackError) -> Self { Self::Pack(v) }
}
impl From<ExpertBlobError> for ComponentReadError {
    fn from(v: ExpertBlobError) -> Self { Self::Blob(v) }
}

/// An immutable view into a shared expert allocation. Eviction from the
/// cache never invalidates this handle. External handles require separate
/// memory accounting beyond the cache-membership budget.
#[derive(Clone)]
pub struct SharedTensorComponent {
    backing: Arc<[u8]>,
    offset: usize,
    length: usize,
}
impl SharedTensorComponent {
    pub fn as_bytes(&self) -> &[u8] {
        &self.backing[self.offset..self.offset + self.length]
    }
    pub fn len(&self) -> usize { self.length }
    pub fn is_empty(&self) -> bool { self.length == 0 }
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.backing, &other.backing)
    }
}

pub struct VerifiedExpertComponentReader {
    reader: CachedExpertBlobReader,
    // Byte offsets within the corresponding expert's cached allocation.
    components: HashMap<(u32, u32, String), (usize, usize)>,
}

impl VerifiedExpertComponentReader {
    pub fn open(
        blob: &Path,
        index: &Path,
        integrity: &Path,
        component_map: &Path,
        cache_budget_bytes: u64,
        cache_policy: ExpertCachePolicy,
    ) -> Result<Self, ComponentReadError> {
        // This verifies all expert CRC32s and complete, gap-free component
        // coverage before any read handle becomes available.
        let (table, manifest) = verify_component_expert_sidecar(
            blob, index, integrity, component_map,
        )?;
        let mut keys = HashMap::with_capacity(manifest.components.len());
        let mut range_idx = 0usize;
        for component in manifest.components {
            while range_idx < table.ranges.len()
                && (table.ranges[range_idx].layer, table.ranges[range_idx].expert_id)
                    < (component.layer, component.expert_id) {
                range_idx += 1;
            }
            let range = table.ranges.get(range_idx).ok_or_else(|| {
                ComponentReadError::Invalid("component lacks indexed expert".into())
            })?;
            if (range.layer,range.expert_id) != (component.layer,component.expert_id) {
                return Err(ComponentReadError::Invalid(
                    "component key does not match an indexed expert".into(),
                ));
            }
            let relative = component.byte_offset.checked_sub(range.byte_offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| ComponentReadError::Invalid(
                    "component-relative offset exceeds host address space".into(),
                ))?;
            let length = usize::try_from(component.byte_length)
                .map_err(|_| ComponentReadError::Invalid(
                    "component length exceeds host address space".into(),
                ))?;
            let expert_size = usize::try_from(range.byte_length)
                .map_err(|_| ComponentReadError::Invalid(
                    "expert length exceeds host address space".into(),
                ))?;
            if relative.checked_add(length).is_none_or(|end| end > expert_size) {
                return Err(ComponentReadError::Invalid(
                    "component exceeds cached expert range".into(),
                ));
            }
            let key = (component.layer, component.expert_id, component.tensor_name);
            if keys.insert(key, (relative,length)).is_some() {
                return Err(ComponentReadError::Invalid("duplicate component key".into()));
            }
        }
        Ok(Self {
            reader: CachedExpertBlobReader::open(
                blob, table, cache_budget_bytes, cache_policy,
            )?,
            components: keys,
        })
    }

    pub fn read_component(
        &self,
        layer: u32,
        expert_id: u32,
        tensor_name: &str,
    ) -> Result<SharedTensorComponent, ComponentReadError> {
        let (offset, length) = *self.components
            .get(&(layer, expert_id, tensor_name.to_owned()))
            .ok_or_else(|| ComponentReadError::NotFound {
                layer, expert_id, tensor_name: tensor_name.to_owned(),
            })?;
        let backing = self.reader.read_expert_shared(layer, expert_id)?;
        if offset.checked_add(length).is_none_or(|end| end > backing.len()) {
            return Err(ComponentReadError::Invalid(
                "cached expert is shorter than validated component".into(),
            ));
        }
        Ok(SharedTensorComponent { backing, offset, length })
    }

    /// Copy one verified component into a reusable caller-owned buffer.
    /// Reject mismatched lengths without reading expert data.
    pub fn read_component_into(
        &self, layer: u32, expert_id: u32, tensor_name: &str,
        dst: &mut [u8],
    ) -> Result<(), ComponentReadError> {
        let &(_, expected) = self.components
            .get(&(layer, expert_id, tensor_name.to_owned()))
            .ok_or_else(|| ComponentReadError::NotFound {
                layer, expert_id, tensor_name: tensor_name.to_owned(),
            })?;
        if dst.len() != expected {
            return Err(ComponentReadError::Invalid(format!(
                "destination buffer has {} bytes; expected {expected}", dst.len(),
            )));
        }
        let view = self.read_component(layer, expert_id, tensor_name)?;
        dst.copy_from_slice(view.as_bytes());
        Ok(())
    }

    /// Return the exact byte length from the verified sidecar manifest
    /// without accessing the packed expert payload.
    pub fn component_byte_length(
        &self, layer: u32, expert_id: u32, tensor_name: &str,
    ) -> Option<usize> {
        self.components.get(&(layer, expert_id, tensor_name.to_owned()))
            .map(|&(_, length)| length)
    }

    pub fn cache_stats(&self) -> ExpertCacheStats { self.reader.cache_stats() }
    pub fn clear_cache(&self) -> usize { self.reader.clear_cache() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_models::{GgufExpertSourceMap, ExpertTensorComponent};
    use crate::expert_sidecar::write_component_expert_sidecar;

    #[test]
    fn reads_exact_tensor_components_without_copying_cached_expert() {
        let root = std::env::temp_dir().join(format!(
            "kiln-components-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("source.gguf");
        let blob=root.join("experts.bin");
        let index=root.join("index.json");
        let integrity=root.join("crc.json");
        let map=root.join("components.json");
        let bytes: Vec<u8>=(0..200u8).collect();
        std::fs::write(&source,&bytes).unwrap();
        let input = GgufExpertSourceMap {
            format_version:1,source_size_bytes:bytes.len() as u64,
            components:vec![
                ExpertTensorComponent { layer:0,expert_id:0,tensor_name:"down".into(),
                    source_byte_offset:40,byte_length:20 },
                ExpertTensorComponent { layer:0,expert_id:0,tensor_name:"up".into(),
                    source_byte_offset:90,byte_length:30 },
                ExpertTensorComponent { layer:0,expert_id:1,tensor_name:"gate".into(),
                    source_byte_offset:150,byte_length:25 },
            ],
        };
        write_component_expert_sidecar(
            &source,input,&blob,&index,&integrity,&map,
        ).unwrap();
        let reader=VerifiedExpertComponentReader::open(
            &blob,&index,&integrity,&map,50,ExpertCachePolicy::Lru,
        ).unwrap();
        let down=reader.read_component(0,0,"down").unwrap();
        let up=reader.read_component(0,0,"up").unwrap();
        assert_eq!(down.as_bytes(),&bytes[40..60]);
        assert_eq!(up.as_bytes(),&bytes[90..120]);
        assert!(down.shares_allocation_with(&up));
        assert_eq!(reader.cache_stats().used_bytes,50);
        let mut reuse=[0u8;20];
        reader.read_component_into(0,0,"down",&mut reuse).unwrap();
        assert_eq!(reuse.as_slice(),&bytes[40..60]);
        assert_eq!(reader.component_byte_length(0,0,"down"),Some(20));
        let before=reader.cache_stats();
        assert!(reader.read_component_into(0,0,"down",&mut [0u8;4]).is_err());
        assert_eq!(reader.cache_stats().hits,before.hits);
        assert_eq!(reader.cache_stats().misses,before.misses);
        let gate=reader.read_component(0,1,"gate").unwrap();
        assert_eq!(gate.as_bytes(),&bytes[150..175]);
        assert_eq!(down.as_bytes(),&bytes[40..60]); // survives eviction
        assert_eq!(reader.cache_stats().used_bytes,25);
        assert!(matches!(reader.read_component(0,1,"unknown"),
            Err(ComponentReadError::NotFound { .. })));
        for file in [&source,&blob,&index,&integrity,&map] {
            std::fs::remove_file(file).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn corrupt_sidecar_fails_before_reader_is_opened() {
        let root = std::env::temp_dir().join(format!(
            "kiln-components-corrupt-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("source");
        let blob=root.join("blob");
        let index=root.join("index");
        let integrity=root.join("integrity");
        let map=root.join("map");
        std::fs::write(&source,[7u8;32]).unwrap();
        write_component_expert_sidecar(&source,GgufExpertSourceMap {
            format_version:1,source_size_bytes:32,
            components:vec![ExpertTensorComponent {
                layer:0,expert_id:0,tensor_name:"weight".into(),
                source_byte_offset:0,byte_length:32,
            }],
        },&blob,&index,&integrity,&map).unwrap();
        std::fs::write(&blob,[9u8;32]).unwrap();
        assert!(VerifiedExpertComponentReader::open(
            &blob,&index,&integrity,&map,128,ExpertCachePolicy::Lru
        ).is_err());
        for file in [&source,&blob,&index,&integrity,&map] {
            std::fs::remove_file(file).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }
}
