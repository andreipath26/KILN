//! Byte-pair encoding tokenizer.
//!
//! See docs/tokenizer-design.md for the design. Reads vocabulary and
//! merge rules from GGUF metadata, encodes text to token IDs, decodes
//! token IDs back to text.

use std::collections::HashMap;

use crate::gguf::{GgufFile, GgufValue};

/// Errors from the tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    MissingMetadata(String),
    MalformedMerge(String),
    UnknownToken(u32),
    InvalidType(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::MissingMetadata(key) => {
                write!(f, "missing metadata key: {}", key)
            }
            TokenizerError::MalformedMerge(s) => {
                write!(f, "malformed merge rule: {}", s)
            }
            TokenizerError::UnknownToken(id) => {
                write!(f, "unknown token id: {}", id)
            }
            TokenizerError::InvalidType(key) => {
                write!(f, "metadata key has wrong type: {}", key)
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// A byte-pair encoding tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string to token id.
    vocab: HashMap<String, u32>,
    /// Merge rule: pair of strings to merge rank.
    merges: HashMap<(String, String), u32>,
    /// Inverse vocabulary: token id to token string.
    inverse_vocab: Vec<String>,
    /// Special token IDs.
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    /// Whether to add BOS at the start of encode.
    pub add_bos_token: bool,
    /// Whether to add EOS at the end of encode.
    pub add_eos_token: bool,
}

impl BpeTokenizer {
    /// Build a tokenizer from GGUF metadata.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, TokenizerError> {
        // Read the vocabulary.
        let tokens_value = file
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| TokenizerError::MissingMetadata(
                "tokenizer.ggml.tokens".to_string()
            ))?;
        let vocab_strings = match tokens_value {
            GgufValue::Array { values, .. } => {
                let mut v = Vec::with_capacity(values.len());
                for (i, val) in values.iter().enumerate() {
                    match val {
                        GgufValue::String(s) => v.push(s.clone()),
                        _ => return Err(TokenizerError::InvalidType(format!(
                            "tokenizer.ggml.tokens[{}]", i
                        ))),
                    }
                }
                v
            }
            _ => return Err(TokenizerError::InvalidType(
                "tokenizer.ggml.tokens".to_string()
            )),
        };

        // Build vocab and inverse vocab.
        let mut vocab = HashMap::with_capacity(vocab_strings.len());
        for (i, s) in vocab_strings.iter().enumerate() {
            vocab.insert(s.clone(), i as u32);
        }
        let inverse_vocab = vocab_strings;

        // Read the merges.
        let mut merges = HashMap::new();
        if let Some(merges_value) = file.metadata.get("tokenizer.ggml.merges") {
            if let GgufValue::Array { values, .. } = merges_value {
                for (rank, val) in values.iter().enumerate() {
                    let s = match val {
                        GgufValue::String(s) => s,
                        _ => continue,
                    };
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next().unwrap_or("").to_string();
                    let b = parts.next().unwrap_or("").to_string();
                    if a.is_empty() || b.is_empty() {
                        return Err(TokenizerError::MalformedMerge(s.clone()));
                    }
                    merges.insert((a, b), rank as u32);
                }
            }
        }

        // Read special token IDs.
        let bos_token_id = file.metadata.get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_token_id = file.metadata.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let padding_token_id = file.metadata.get("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.as_u32());

        let add_bos_token = file.metadata.get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let add_eos_token = file.metadata.get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            vocab,
            merges,
            inverse_vocab,
            bos_token_id,
            eos_token_id,
            padding_token_id,
            add_bos_token,
            add_eos_token,
        })
    }

    /// Encode text to token IDs. Uses the naive O(n^2) algorithm.
    /// Suitable for inputs under 1000 characters.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Split text into characters. Each character is a candidate token.
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        if tokens.is_empty() {
            return Vec::new();
        }

        // Iteratively merge the pair with the lowest merge rank.
        loop {
            let mut best_rank: Option<u32> = None;
            let mut best_pos: Option<usize> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    match best_rank {
                        None => {
                            best_rank = Some(rank);
                            best_pos = Some(i);
                        }
                        Some(current) if rank < current => {
                            best_rank = Some(rank);
                            best_pos = Some(i);
                        }
                        _ => {}
                    }
                }
            }

            let pos = match best_pos {
                Some(p) => p,
                None => break,
            };

            // Merge the pair at pos.
            let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
            tokens[pos] = merged;
            tokens.remove(pos + 1);
        }

        // Map each remaining string to its token ID.
        let mut ids = Vec::with_capacity(tokens.len());
        for t in tokens {
            if let Some(&id) = self.vocab.get(&t) {
                ids.push(id);
            } else {
                // Character not in vocab. Try byte-level fallback by
                // encoding the UTF-8 bytes individually. Most BPE
                // vocabularies include single-byte tokens.
                for byte in t.as_bytes() {
                    let byte_str = String::from_utf8_lossy(&[*byte]).to_string();
                    if let Some(&id) = self.vocab.get(&byte_str) {
                        ids.push(id);
                    }
                    // If even the byte is not in the vocab, skip it.
                    // That is a model quality issue, not a tokenizer bug.
                }
            }
        }

        ids
    }

    /// Encode text and optionally add BOS and EOS.
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                ids.push(bos);
            }
        }
        ids.extend(self.encode(text));
        if add_eos {
            if let Some(eos) = self.eos_token_id {
                ids.push(eos);
            }
        }
        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &id in tokens {
            if Some(id) == self.bos_token_id
                || Some(id) == self.eos_token_id
                || Some(id) == self.padding_token_id
            {
                continue;
            }
            if let Some(s) = self.inverse_vocab.get(id as usize) {
                out.push_str(s);
            }
        }
        out
    }

    /// Look up the string for a token ID.
    pub fn token_to_str(&self, token: u32) -> Option<&str> {
        self.inverse_vocab.get(token as usize).map(|s| s.as_str())
    }

    /// Look up the ID for a token string.
    pub fn str_to_token(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// The size of the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inverse_vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufValue, GgufValueType};
    use std::collections::HashMap;

    /// Build a fake GgufFile for testing. This bypasses the loader
    /// because the loader requires a real file. The tokenizer only
    /// needs the metadata map.
    fn make_fake_gguf(
        vocab: Vec<&str>,
        merges: Vec<&str>,
    ) -> (HashMap<String, GgufValue>, BpeTokenizer) {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array {
                element_type: GgufValueType::String,
                values: vocab.iter().map(|s| GgufValue::String(s.to_string())).collect(),
            },
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array {
                element_type: GgufValueType::String,
                values: merges.iter().map(|s| GgufValue::String(s.to_string())).collect(),
            },
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Uint32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Uint32(2),
        );
        // We cannot construct a GgufFile easily (it has an mmap). The
        // tokenizer's from_gguf only reads metadata, so we test the
        // internal logic by building a tokenizer manually.
        let mut vocab_map = HashMap::new();
        for (i, s) in vocab.iter().enumerate() {
            vocab_map.insert(s.to_string(), i as u32);
        }
        let mut merge_map = HashMap::new();
        for (rank, s) in merges.iter().enumerate() {
            let mut parts = s.splitn(2, ' ');
            let a = parts.next().unwrap_or("").to_string();
            let b = parts.next().unwrap_or("").to_string();
            merge_map.insert((a, b), rank as u32);
        }
        // Find the actual IDs of the special tokens in the vocab.
        let bos_id = vocab.iter().position(|s| *s == "<bos>").map(|i| i as u32);
        let eos_id = vocab.iter().position(|s| *s == "<eos>").map(|i| i as u32);
        let tok = BpeTokenizer {
            vocab: vocab_map,
            merges: merge_map,
            inverse_vocab: vocab.iter().map(|s| s.to_string()).collect(),
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            padding_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
        };
        (metadata, tok)
    }

    #[test]
    fn encode_decode_round_trip() {
        // Simple vocab: single chars a b c plus merged "ab" and "abc"
        let (_md, tok) = make_fake_gguf(
            vec!["a", "b", "c", "ab", "abc", "<bos>", "<eos>"],
            vec!["a b", "ab c"],
        );
        let ids = tok.encode("abc");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.token_to_str(ids[0]).unwrap(), "abc");
        let text = tok.decode(&ids);
        assert_eq!(text, "abc");
    }

    #[test]
    fn encode_without_merges() {
        let (_md, tok) = make_fake_gguf(
            vec!["a", "b", "c", "<bos>", "<eos>"],
            vec![],
        );
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![0, 1, 2], "encode produced wrong ids");
        assert_eq!(tok.decode(&ids), "abc");
    }

    #[test]
    fn encode_with_special_tokens() {
        let (_md, tok) = make_fake_gguf(
            vec!["h", "i", "<bos>", "<eos>"],
            vec![],
        );
        // h=0, i=1, <bos>=2, <eos>=3
        let ids = tok.encode_with_special("hi", true, true);
        assert_eq!(ids, vec![2, 0, 1, 3]); // <bos>, h, i, <eos>
    }

    #[test]
    fn decode_skips_special_tokens() {
        let (_md, tok) = make_fake_gguf(
            vec!["h", "i", "<bos>", "<eos>"],
            vec![],
        );
        // h=0, i=1, <bos>=2, <eos>=3
        let ids = vec![2, 0, 1, 3]; // <bos>, h, i, <eos>
        let text = tok.decode(&ids);
        assert_eq!(text, "hi");
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let (_md, tok) = make_fake_gguf(vec!["a", "<bos>", "<eos>"], vec![]);
        let ids = tok.encode("");
        assert!(ids.is_empty());
        assert_eq!(tok.decode(&ids), "");
    }
}
