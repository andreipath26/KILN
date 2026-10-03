//! The chat loop.
//!
//! See docs/chat-loop-design.md for the design. The generation loop
//! that turns a single matmul into a conversation.

use kiln_models::tokenizer::{BpeTokenizer, TokenizerError};

/// A forward pass that produces logits for the next token.
pub trait Forward: Send {
    /// Given a token sequence, return the logits for the next token.
    /// The logits vector has one value per vocabulary entry.
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32>;

    /// The vocabulary size. Default is derived from the first forward
    /// call. Override this to make the sampler know the size up front.
    fn vocab_size(&self) -> usize {
        0
    }
}

/// A mock forward pass for testing. Returns fixed logits.
pub struct MockForward {
    logits: Vec<f32>,
    call_count: usize,
    /// After this many calls, switch to a different logits vector.
    /// Used to test EOS handling.
    switch_at: Option<usize>,
    switched_logits: Option<Vec<f32>>,
}

impl MockForward {
    pub fn new(logits: Vec<f32>) -> Self {
        Self {
            logits,
            call_count: 0,
            switch_at: None,
            switched_logits: None,
        }
    }

    pub fn with_switch(mut self, at: usize, logits: Vec<f32>) -> Self {
        self.switch_at = Some(at);
        self.switched_logits = Some(logits);
        self
    }
}

impl Forward for MockForward {
    fn forward(&mut self, _tokens: &[u32]) -> Vec<f32> {
        self.call_count += 1;
        if let Some(at) = self.switch_at {
            if self.call_count > at {
                if let Some(ref switched) = self.switched_logits {
                    return switched.clone();
                }
            }
        }
        self.logits.clone()
    }

    fn vocab_size(&self) -> usize {
        self.logits.len()
    }
}

/// A synthetic forward pass for pipeline testing. Uses a seeded
/// xorshift to produce pseudo-random logits from the token sequence.
pub struct SyntheticForward {
    vocab_size: usize,
    seed: u64,
}

impl SyntheticForward {
    pub fn new(vocab_size: usize, seed: u64) -> Self {
        Self { vocab_size, seed }
    }
}

impl Forward for SyntheticForward {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        // Hash the token sequence and the seed to get a reproducible
        // logits vector.
        let mut h: u64 = self.seed;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001B3);
        }
        let mut state = h | 1;
        let mut logits = Vec::with_capacity(self.vocab_size);
        for _ in 0..self.vocab_size {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to a float in [-1, 1].
            let v = ((state as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
            logits.push(v);
        }
        logits
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// The sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Always pick the highest-probability token.
    Greedy,
    /// Sample from the top-k most likely tokens with the given
    /// temperature.
    TopK { k: usize, temperature: f32 },
    /// Sample from tokens whose cumulative probability exceeds the
    /// given threshold.
    TopP { p: f32, temperature: f32 },
}

/// A seeded pseudo-random sampler.
pub struct Sampler {
    strategy: SamplingStrategy,
    state: u64,
}

impl Sampler {
    pub fn new(strategy: SamplingStrategy, seed: u64) -> Self {
        Self {
            strategy,
            state: seed | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Pick a token ID from the logits.
    pub fn sample(&mut self, logits: &[f32]) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }
        match self.strategy {
            SamplingStrategy::Greedy => {
                let mut best_idx = 0usize;
                let mut best_val = logits[0];
                for (i, &v) in logits.iter().enumerate().skip(1) {
                    if v > best_val {
                        best_val = v;
                        best_idx = i;
                    }
                }
                Some(best_idx as u32)
            }
            SamplingStrategy::TopK { k, temperature } => {
                self.sample_top_k(logits, k, temperature)
            }
            SamplingStrategy::TopP { p, temperature } => {
                self.sample_top_p(logits, p, temperature)
            }
        }
    }

    fn sample_top_k(&mut self, logits: &[f32], k: usize, temperature: f32) -> Option<u32> {
        let k = k.max(1).min(logits.len());
        // Apply temperature.
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
        // Find the top-k indices.
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(k);
        // Softmax over the top-k.
        let max_val = scaled[indices[0]];
        let mut probs: Vec<f32> = indices.iter().map(|&i| (scaled[i] - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // Sample.
        let r = self.next_f32();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return Some(indices[i] as u32);
            }
        }
        Some(indices[indices.len() - 1] as u32)
    }

    fn sample_top_p(&mut self, logits: &[f32], p: f32, temperature: f32) -> Option<u32> {
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Keep the smallest set whose cumulative probability exceeds p.
        let mut cum = 0.0;
        let mut keep = 0;
        for (i, (_, prob)) in probs.iter().enumerate() {
            cum += prob;
            keep = i + 1;
            if cum >= p {
                break;
            }
        }
        probs.truncate(keep);
        // Renormalize.
        let sum2: f32 = probs.iter().map(|&(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum2;
        }
        // Sample.
        let r = self.next_f32();
        let mut cum2 = 0.0;
        for &(idx, prob) in probs.iter() {
            cum2 += prob;
            if r < cum2 {
                return Some(idx as u32);
            }
        }
        Some(probs.last().map(|&(i, _)| i as u32).unwrap_or(0))
    }
}

/// Errors from the chat session.
#[derive(Debug)]
pub enum ChatError {
    Tokenizer(TokenizerError),
    EmptyVocabulary,
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Tokenizer(e) => write!(f, "tokenizer error: {}", e),
            ChatError::EmptyVocabulary => write!(f, "forward pass returned empty logits"),
        }
    }
}

impl std::error::Error for ChatError {}

impl From<TokenizerError> for ChatError {
    fn from(e: TokenizerError) -> Self {
        ChatError::Tokenizer(e)
    }
}

/// A chat session.
pub struct ChatSession {
    pub tokenizer: BpeTokenizer,
    pub forward: Box<dyn Forward>,
    pub sampler: Sampler,
    pub history: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,
}

impl ChatSession {
    pub fn new(
        tokenizer: BpeTokenizer,
        forward: Box<dyn Forward>,
        sampler: Sampler,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            tokenizer,
            forward,
            sampler,
            history: Vec::new(),
            max_new_tokens,
            eos_token_id,
        }
    }

    /// Generate a response to the prompt. Returns the full text.
    pub fn generate(&mut self, prompt: &str) -> Result<String, ChatError> {
        let mut out = String::new();
        self.generate_streaming(prompt, &mut |s| out.push_str(s))?;
        Ok(out)
    }

    /// Generate a response, calling `sink` for each new piece of text.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        sink: &mut dyn FnMut(&str),
    ) -> Result<(), ChatError> {
        // Encode the prompt and append to history.
        let prompt_tokens = self.tokenizer.encode(prompt);
        self.history.extend(prompt_tokens);

        for _ in 0..self.max_new_tokens {
            let logits = self.forward.forward(&self.history);
            if logits.is_empty() {
                return Err(ChatError::EmptyVocabulary);
            }
            let token = match self.sampler.sample(&logits) {
                Some(t) => t,
                None => break,
            };
            self.history.push(token);
            if Some(token) == self.eos_token_id {
                break;
            }
            if let Some(s) = self.tokenizer.token_to_str(token) {
                sink(s);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_tokenizer(vocab: Vec<&str>, merges: Vec<&str>) -> BpeTokenizer {
        let vocab_strings: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let merges_strings: Vec<String> = merges.iter().map(|s| s.to_string()).collect();
        let bos_id = vocab.iter().position(|s| *s == "<bos>").map(|i| i as u32);
        let eos_id = vocab.iter().position(|s| *s == "<eos>").map(|i| i as u32);
        BpeTokenizer::from_parts(vocab_strings, merges_strings, bos_id, eos_id)
    }

    fn make_session(vocab: Vec<&str>, logits: Vec<f32>, max_new: usize) -> ChatSession {
        let tok = make_tokenizer(vocab.clone(), vec![]);
        let fwd = MockForward::new(logits);
        let sampler = Sampler::new(SamplingStrategy::Greedy, 42);
        let eos = vocab.iter().position(|s| *s == "<eos>").map(|i| i as u32);
        ChatSession::new(tok, Box::new(fwd), sampler, max_new, eos)
    }

    #[test]
    fn deterministic_greedy() {
        // Two sessions with the same seed and same prompt should
        // produce identical output.
        let vocab = vec!["a", "b", "c", "<bos>", "<eos>"];
        let logits = vec![0.1, 0.5, 0.3, -1.0, -1.0]; // greedy picks index 1 = "b"
        let mut s1 = make_session(vocab.clone(), logits.clone(), 5);
        let mut s2 = make_session(vocab.clone(), logits.clone(), 5);
        let r1 = s1.generate("a").unwrap();
        let r2 = s2.generate("a").unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1, "bbbbb");
    }

    #[test]
    fn max_new_tokens_respected() {
        let vocab = vec!["a", "b", "c", "<bos>", "<eos>"];
        let logits = vec![0.1, 0.5, 0.3, -1.0, -1.0]; // picks "b"
        let mut s = make_session(vocab, logits, 3);
        let r = s.generate("a").unwrap();
        assert_eq!(r, "bbb");
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn eos_stops_generation() {
        // Logits vector with highest value at index 4 = "<eos>".
        let vocab = vec!["a", "b", "c", "<bos>", "<eos>"];
        let logits = vec![-1.0, -1.0, -1.0, -1.0, 1.0];
        let mut s = make_session(vocab, logits, 10);
        let r = s.generate("a").unwrap();
        // First token is EOS, so output is empty.
        assert_eq!(r, "");
    }

    #[test]
    fn empty_logits_errors() {
        let vocab = vec!["a", "b", "c", "<bos>", "<eos>"];
        let logits = vec![];
        let mut s = make_session(vocab, logits, 5);
        let r = s.generate("a");
        assert!(matches!(r, Err(ChatError::EmptyVocabulary)));
    }
}
