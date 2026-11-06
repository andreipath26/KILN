//! CPU-only draftless n-gram speculation candidate generator.
//!
//! A draft is a *proposal*, never an inference result. The target model must
//! verify every proposed token before KILN is allowed to emit it.

/// Maximum number of draft tokens per verification pass. This intentionally
/// stays small enough for the Tier 0 CPU verification experiments.
pub const MAX_DRAFT_TOKENS: usize = 16;

/// A bounded history-matching drafter. No additional model or network I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramDraftConfig {
    /// Exact trailing token sequence to match (typically 2 or 3).
    pub ngram: usize,
    /// At most this many preceding tokens are searched.
    pub search_window: usize,
    /// Maximum number of proposed continuation tokens.
    pub max_draft: usize,
}

impl Default for NgramDraftConfig {
    fn default() -> Self {
        Self { ngram: 3, search_window: 8192, max_draft: 8 }
    }
}

impl NgramDraftConfig {
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.ngram == 0 || self.ngram > 16 { return Err("ngram must be 1..=16"); }
        if self.search_window == 0 { return Err("search_window must be positive"); }
        if self.max_draft == 0 || self.max_draft > MAX_DRAFT_TOKENS {
            return Err("max_draft must be 1..=16");
        }
        Ok(self)
    }
}

/// Search the latest prefix match in a bounded window, choosing the most
/// recent occurrence deterministically. Tokens beyond the existing history
/// are never invented. EOS and tokens after it are not proposed.
pub fn draft_ngram(history: &[i32], cfg: NgramDraftConfig, eos: Option<i32>) -> Vec<i32> {
    if cfg.validate().is_err() || history.len() <= cfg.ngram { return Vec::new(); }
    let n = cfg.ngram;
    let suffix_start = history.len() - n;
    let earliest = suffix_start.saturating_sub(cfg.search_window);
    let suffix = &history[suffix_start..];
    // The source match must have at least one successor already recorded.
    for candidate in (earliest..suffix_start).rev() {
        if candidate + n >= history.len() { continue; }
        if &history[candidate..candidate + n] != suffix { continue; }
        let start = candidate + n;
        // A draft is restricted to recorded tokens, including overlapping
        // suffixes, with EOS always excluded from the candidate.
        let end = history.len().min(start.saturating_add(cfg.max_draft));
        let mut out = Vec::with_capacity(end - start);
        for &token in &history[start..end] {
            if Some(token) == eos { break; }
            out.push(token);
        }
        return out;
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_most_recent_matching_suffix() {
        let cfg = NgramDraftConfig { ngram: 2, search_window: 100, max_draft: 4 };
        let history = [5, 6, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2];
        assert_eq!(draft_ngram(&history, cfg, None), vec![20, 21, 1, 2]);
    }

    #[test]
    fn never_proposes_unknown_tokens_or_eos() {
        let cfg = NgramDraftConfig { ngram: 2, search_window: 100, max_draft: 8 };
        assert_eq!(draft_ngram(&[1, 2, 3, 4, 1, 2], cfg, Some(4)), vec![3]);
        assert!(draft_ngram(&[1, 2, 3], cfg, None).is_empty());
    }

    #[test]
    fn window_and_invalid_configuration_fail_closed() {
        let history = [7, 8, 9, 7, 8];
        assert!(draft_ngram(&history, NgramDraftConfig { ngram: 2, search_window: 1, max_draft: 5 }, None).is_empty());
        assert!(draft_ngram(&history, NgramDraftConfig { ngram: 0, search_window: 100, max_draft: 5 }, None).is_empty());
        assert!(draft_ngram(&history, NgramDraftConfig { ngram: 2, search_window: 100, max_draft: 17 }, None).is_empty());
    }

    #[test]
    fn overlapping_recent_match_stays_within_known_history() {
        let cfg = NgramDraftConfig { ngram: 2, search_window: 100, max_draft: 8 };
        assert_eq!(draft_ngram(&[3, 3, 3, 3], cfg, None), vec![3]);
    }
}
