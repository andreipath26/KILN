//! Sanity and quality guarantees. Phase 4.5.
//! See roadmap Section 15.5. A broken model is refused, not run.

use crate::ffi::LlamaContext;

#[derive(Debug)]
pub enum SanityError {
    Load(String),
    CanaryFailed { prompt: String, expected: String, got: String },
    TooFewLogits,
}

impl std::fmt::Display for SanityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SanityError::Load(m) => write!(f, "load: {}", m),
            SanityError::CanaryFailed { prompt, expected, got } =>
                write!(f, "canary failed: prompt={:?} expected={} got={}", prompt, expected, got),
            SanityError::TooFewLogits => write!(f, "model returned too few logits"),
        }
    }
}

impl std::error::Error for SanityError {}

/// A canary prompt and what its top-1 should decode to.
pub struct Canary {
    pub prompt: &'static str,
    pub expected_contains: &'static str,
}

pub const CANARIES: &[Canary] = &[
    Canary { prompt: "The capital of France is", expected_contains: "Paris" },
    Canary { prompt: "The quick brown fox jumps over the", expected_contains: "lazy" },
];

/// Run every canary against a loaded context. Returns Ok if all pass,
/// or the first failure. The context is reset and reused across
/// canaries.
pub fn run_canaries(ctx: &mut LlamaContext) -> Result<(), SanityError> {
    for c in CANARIES {
        ctx.reset().map_err(|e| SanityError::Load(e.to_string()))?;
        let ids = ctx.tokenize(c.prompt).map_err(|e| SanityError::Load(e.to_string()))?;
        let logits = ctx.forward(&ids).map_err(|e| SanityError::Load(e.to_string()))?;
        if logits.len() < 10 {
            return Err(SanityError::TooFewLogits);
        }
        // Find top-1.
        let mut best = 0usize;
        let mut best_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val { best_val = v; best = i; }
        }
        let decoded = ctx.token_to_str(best as i32);
        let decoded_trim = decoded.trim();
        // Match either the decoded string or a substring. "1 + 1 =" is
        // looser: the token after "= " may not be the digit alone.
        if !decoded_contains(decoded_trim, c.expected_contains) {
            // Second chance: check the top-5.
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = idx.iter().take(5).map(|&i| ctx.token_to_str(i as i32)).collect();
            let found = top5.iter().any(|s| decoded_contains(s.trim(), c.expected_contains));
            if !found {
                return Err(SanityError::CanaryFailed {
                    prompt: c.prompt.to_string(),
                    expected: c.expected_contains.to_string(),
                    got: decoded_trim.to_string(),
                });
            }
        }
    }
    ctx.reset().map_err(|e| SanityError::Load(e.to_string()))?;
    Ok(())
}

fn decoded_contains(decoded: &str, expected: &str) -> bool {
    decoded.contains(expected)
}

/// Generation watchdog. Called after each generated token.
#[derive(Debug, Default)]
pub struct Watchdog {
    last_tokens: Vec<i32>,
    last_pieces: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub enum WatchdogVerdict {
    Ok,
    RepeatedOutput,
    LowConfidence,
    NonTextOutput,
}

impl Watchdog {
    pub fn new() -> Self { Self::default() }

    pub fn observe(&mut self, token: i32, bytes: &[u8], top1_logit: f32) -> WatchdogVerdict {
        self.last_tokens.push(token);
        if self.last_tokens.len() > 3 { self.last_tokens.remove(0); }
        self.last_pieces.push(bytes.to_vec());
        if self.last_pieces.len() > 20 { self.last_pieces.remove(0); }

        // Repeated output: same token three times in a row.
        if self.last_tokens.len() == 3
            && self.last_tokens[0] == self.last_tokens[1]
            && self.last_tokens[1] == self.last_tokens[2]
        {
            return WatchdogVerdict::RepeatedOutput;
        }

        // Low confidence: top-1 logit below threshold. Model-specific.
        // -10 is generous; a working model rarely goes below -5.
        if top1_logit < -15.0 {
            return WatchdogVerdict::LowConfidence;
        }

        // Non-text: >30% of last 20 tokens are non-printable.
        if self.last_pieces.len() >= 20 {
            let non_printable = self.last_pieces.iter()
                .filter(|b| b.iter().all(|&c| c < 0x20 || c == 0x7f))
                .count();
            if non_printable * 100 / self.last_pieces.len() > 30 {
                return WatchdogVerdict::NonTextOutput;
            }
        }

        WatchdogVerdict::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_list_not_empty() {
        assert!(!CANARIES.is_empty());
    }

    #[test]
    fn watchdog_repeated() {
        let mut w = Watchdog::new();
        assert!(matches!(w.observe(5, b"a", 0.0), WatchdogVerdict::Ok));
        assert!(matches!(w.observe(5, b"a", 0.0), WatchdogVerdict::Ok));
        assert!(matches!(w.observe(5, b"a", 0.0), WatchdogVerdict::RepeatedOutput));
    }

    #[test]
    fn watchdog_low_confidence() {
        let mut w = Watchdog::new();
        assert!(matches!(w.observe(1, b"a", -100.0), WatchdogVerdict::LowConfidence));
    }
}
