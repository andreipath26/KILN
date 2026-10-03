//! Transformer configuration read from GGUF metadata.
//!
//! See docs/transformer-design.md for the design.

use kiln_models::gguf::{GgufFile, GgufValue};

/// Errors from config loading.
#[derive(Debug)]
pub enum ConfigError {
    MissingKey(String),
    WrongType(String),
    UnknownArchitecture(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingKey(k) => write!(f, "missing metadata key: {}", k),
            ConfigError::WrongType(k) => write!(f, "metadata key has wrong type: {}", k),
            ConfigError::UnknownArchitecture(a) => {
                write!(f, "unknown architecture: {}", a)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// The transformer configuration.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// The model family, e.g. "qwen2", "llama".
    pub arch: String,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (grouped query attention).
    pub num_kv_heads: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Maximum context length.
    pub context_length: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
}

impl TransformerConfig {
    /// Read the config from a GGUF file.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ConfigError> {
        let arch = file
            .metadata_str("general.architecture")
            .ok_or_else(|| ConfigError::MissingKey("general.architecture".to_string()))?
            .to_string();

        let prefix = format!("{}.", arch);

        let get_u32 = |key: &str| -> Result<usize, ConfigError> {
            let full_key = format!("{}{}", prefix, key);
            file.metadata_u32(&full_key)
                .map(|v| v as usize)
                .ok_or(ConfigError::MissingKey(full_key))
        };

        let get_f32 = |key: &str| -> Result<f32, ConfigError> {
            let full_key = format!("{}{}", prefix, key);
            file.metadata
                .get(&full_key)
                .and_then(|v| v.as_f32())
                .ok_or(ConfigError::MissingKey(full_key))
        };

        let vocab_size = get_u32("vocab_size")?;
        let hidden_size = get_u32("embedding_length")?;
        let num_layers = get_u32("block_count")?;
        let intermediate_size = get_u32("feed_forward_length")?;
        let num_heads = get_u32("attention.head_count")?;

        // num_kv_heads may be absent in older models. Default to num_heads.
        let num_kv_heads = file
            .metadata_u32(&format!("{}attention.head_count_kv", prefix))
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon")?;

        let context_length = file
            .metadata_u32(&format!("{}context_length", prefix))
            .map(|v| v as usize)
            .unwrap_or(2048);

        let rope_theta = file
            .metadata
            .get(&format!("{}rope.freq_base", prefix))
            .and_then(|v| v.as_f32())
            .unwrap_or(10000.0);

        Ok(Self {
            arch,
            vocab_size,
            hidden_size,
            num_layers,
            intermediate_size,
            num_heads,
            num_kv_heads,
            rms_norm_eps,
            context_length,
            rope_theta,
        })
    }

    /// The dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// A short human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{} vocab={} hidden={} layers={} inter={} heads={} kv_heads={} context={}",
            self.arch,
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.intermediate_size,
            self.num_heads,
            self.num_kv_heads,
            self.context_length,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_computed() {
        let cfg = TransformerConfig {
            arch: "test".to_string(),
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            intermediate_size: 128,
            num_heads: 4,
            num_kv_heads: 4,
            rms_norm_eps: 1e-5,
            context_length: 2048,
            rope_theta: 10000.0,
        };
        assert_eq!(cfg.head_dim(), 16);
    }
}
