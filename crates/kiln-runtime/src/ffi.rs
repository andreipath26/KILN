//! Rust wrapper over the KILN llama.cpp shim.
//! The shim lives in cpp/llama_shim/. Rust never sees llama.cpp
//! structs — every call goes through a stable extern "C" ABI.
//! See docs/runtime-loading-design.md.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

#[derive(Debug)]
pub enum LlamaError {
    Load(String),
    Tokenize(String),
    Decode(String),
}

impl std::fmt::Display for LlamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlamaError::Load(m) => write!(f, "load: {}", m),
            LlamaError::Tokenize(m) => write!(f, "tokenize: {}", m),
            LlamaError::Decode(m) => write!(f, "decode: {}", m),
        }
    }
}

impl std::error::Error for LlamaError {}

#[repr(C)]
pub struct LlamaParamsRaw {
    pub n_ctx: c_int,
    pub n_batch: c_int,
    pub n_ubatch: c_int,
    pub n_threads: c_int,
}

extern "C" {
    fn kiln_llama_load(path: *const c_char, params: *const LlamaParamsRaw) -> *mut c_void;
    fn kiln_llama_free(h: *mut c_void);
    fn kiln_llama_n_vocab(h: *mut c_void) -> c_int;
    fn kiln_llama_tokenize(
        h: *mut c_void,
        text: *const c_char,
        text_len: c_int,
        out_tokens: *mut c_int,
        out_cap: c_int,
        add_special: c_int,
        parse_special: c_int,
    ) -> c_int;
    fn kiln_llama_decode(
        h: *mut c_void,
        tokens: *const c_int,
        n_tokens: c_int,
        pos_offset: c_int,
    ) -> c_int;
    fn kiln_llama_logits(h: *mut c_void, out: *mut f32, out_cap: c_int) -> c_int;
    fn kiln_llama_greedy_token(h: *mut c_void) -> c_int;
    fn kiln_llama_token_to_str(
        h: *mut c_void,
        token: c_int,
        buf: *mut c_char,
        buf_len: c_int,
    ) -> c_int;
    fn kiln_llama_reset(h: *mut c_void) -> c_int;
    fn kiln_llama_eos_token(h: *mut c_void) -> c_int;
}

pub struct LlamaContext {
    handle: *mut c_void,
    n_vocab: i32,
    n_past: i32,
    /// Physical reusable batch capacity configured in the C shim.
    batch_capacity: usize,
    /// Maximum token positions available in this unshifted context.
    context_capacity: usize,
}

unsafe impl Send for LlamaContext {}

impl LlamaContext {
    pub fn load(path: &Path) -> Result<Self, LlamaError> {
        Self::load_with(path, None)
    }

    pub fn load_with(path: &Path, params: Option<(i32, i32, i32)>) -> Result<Self, LlamaError> {
        let tuned = params.map(|(ctx, batch, threads)| (ctx, batch, batch.min(512), threads));
        Self::load_with_ubatch(path, tuned)
    }

    /// Explicit physical microbatch tuning for controlled Phase 3.5 experiments.
    pub fn load_with_ubatch(path: &Path, params: Option<(i32, i32, i32, i32)>) -> Result<Self, LlamaError> {
        if let Some((ctx, batch, ubatch, threads)) = params {
            if ctx <= 0 || batch <= 0 || ubatch <= 0 || threads <= 0 || ubatch > batch {
                return Err(LlamaError::Load("invalid native context/batch/microbatch/thread parameters".into()));
            }
        }
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| LlamaError::Load(e.to_string()))?;
        let raw = params.map(|(n_ctx, n_batch, n_ubatch, n_threads)| LlamaParamsRaw {
            n_ctx: n_ctx as c_int,
            n_batch: n_batch as c_int,
            n_ubatch: n_ubatch as c_int,
            n_threads: n_threads as c_int,
        });
        let raw_ptr = raw.as_ref().map(|r| r as *const LlamaParamsRaw).unwrap_or(std::ptr::null());
        let handle = unsafe { kiln_llama_load(cpath.as_ptr(), raw_ptr) };
        if handle.is_null() {
            return Err(LlamaError::Load(format!(
                "shim returned null for {}",
                path.display()
            )));
        }
        let n_vocab = unsafe { kiln_llama_n_vocab(handle) };
        // The shim uses n_batch from the supplied params, or 512 by default.
        // Keep this synchronized with the C shim's allocation contract.
        let batch_capacity = params.map(|(_, n_batch, _, _)| n_batch as usize).unwrap_or(512);
        let context_capacity = params.map(|(ctx, _, _, _)| ctx as usize).unwrap_or(512);
        Ok(Self { handle, n_vocab, n_past: 0, batch_capacity, context_capacity })
    }

    pub fn n_vocab(&self) -> i32 { self.n_vocab }
    pub fn n_past(&self) -> i32 { self.n_past }

    /// Reset the KV cache and the position counter. Use between
    /// turns of a multi-turn conversation on the same context.
    pub fn reset(&mut self) -> Result<(), LlamaError> {
        let rc = unsafe { kiln_llama_reset(self.handle) };
        if rc != 0 {
            return Err(LlamaError::Decode(format!("reset returned {}", rc)));
        }
        self.n_past = 0;
        Ok(())
    }

    /// The EOS token id for this model. -1 if the model has no EOS.
    pub fn eos_token(&self) -> i32 {
        unsafe { kiln_llama_eos_token(self.handle) }
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>, LlamaError> {
        let ctext = CString::new(text).map_err(|e| LlamaError::Tokenize(e.to_string()))?;
        let text_len = c_int::try_from(text.len())
            .map_err(|_| LlamaError::Tokenize("input exceeds the C ABI length limit".to_string()))?;
        let capacity = text.len().checked_add(8)
            .ok_or_else(|| LlamaError::Tokenize("input length overflow".to_string()))?;
        let capacity = c_int::try_from(capacity)
            .map_err(|_| LlamaError::Tokenize("token buffer exceeds the C ABI capacity limit".to_string()))?;
        let mut buf: Vec<i32> = vec![0; capacity as usize];
        let n = unsafe {
            kiln_llama_tokenize(
                self.handle,
                ctext.as_ptr(),
                text_len,
                buf.as_mut_ptr(),
                capacity,
                0,
                1,
            )
        };
        if n < 0 {
            return Err(LlamaError::Tokenize(format!("shim returned {}", n)));
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    pub fn decode(&mut self, tokens: &[i32]) -> Result<(), LlamaError> {
        if tokens.is_empty() {
            return Err(LlamaError::Decode("cannot decode an empty token batch".to_string()));
        }
        let token_count = i32::try_from(tokens.len())
            .map_err(|_| LlamaError::Decode("token batch exceeds the C ABI length limit".to_string()))?;
        let end_position = self.n_past.checked_add(token_count)
            .ok_or_else(|| LlamaError::Decode("token position overflow".to_string()))?;
        if end_position < 0 || end_position as usize > self.context_capacity {
            return Err(LlamaError::Decode(format!(
                "token positions would exceed configured n_ctx (end {}, capacity {})",
                end_position, self.context_capacity
            )));
        }
        if self.batch_capacity == 0 {
            return Err(LlamaError::Decode("native batch capacity must be positive".to_string()));
        }
        // A prompt can exceed n_batch even when it fits in n_ctx. The C shim
        // uses one reusable llama_batch of exactly n_batch entries. Send
        // bounded chunks while retaining KV state and token positions.
        for chunk in tokens.chunks(self.batch_capacity) {
            let count = i32::try_from(chunk.len())
                .map_err(|_| LlamaError::Decode("token batch exceeds the C ABI limit".to_string()))?;
            let rc = unsafe {
                kiln_llama_decode(
                    self.handle,
                    chunk.as_ptr(),
                    count,
                    self.n_past,
                )
            };
            if rc != 0 {
                // Earlier chunks may already be committed to KV state. Never
                // advance past the failed chunk; caller should reset the
                // session before retrying after an error.
                return Err(LlamaError::Decode(format!("shim returned {} (after {} tokens)", rc, self.n_past)));
            }
            self.n_past += count;
        }
        Ok(())
    }

    pub fn logits(&self) -> Result<Vec<f32>, LlamaError> {
        let mut out = vec![0.0f32; self.n_vocab as usize];
        let n = unsafe {
            kiln_llama_logits(self.handle, out.as_mut_ptr(), out.len() as c_int)
        };
        if n < 0 {
            return Err(LlamaError::Decode(format!("logits returned {}", n)));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Forward and select the first maximum token inside the native shim,
    /// avoiding a vocabulary-sized FFI buffer and Rust-side scan per step.
    /// Greedy sampling only; full logits remain available for other samplers.
    pub fn forward_greedy(&mut self, tokens: &[i32]) -> Result<i32, LlamaError> {
        self.decode(tokens)?;
        let token = unsafe { kiln_llama_greedy_token(self.handle) };
        if token < 0 {
            return Err(LlamaError::Decode(format!("native greedy sampling returned {token}")));
        }
        Ok(token)
    }

    pub fn forward(&mut self, tokens: &[i32]) -> Result<Vec<f32>, LlamaError> {
        self.decode(tokens)?;
        self.logits()
    }

    /// Decode one token to raw UTF-8 bytes. Do not lossy-convert here;
    /// a token may carry a partial multi-byte sequence that the next
    /// token completes. The caller buffers bytes and decodes when a
    /// full sequence is available.
    pub fn token_bytes(&self, token: i32) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let n = unsafe {
            kiln_llama_token_to_str(
                self.handle,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
        if n <= 0 { return Vec::new(); }
        buf.truncate(n as usize);
        buf
    }

    /// Backwards-compatible string form. Lossy. Prefer token_bytes.
    pub fn token_to_str(&self, token: i32) -> String {
        String::from_utf8_lossy(&self.token_bytes(token)).to_string()
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { kiln_llama_free(self.handle); }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_decode_batch_is_rejected_before_ffi() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 0,
            batch_capacity: 512,
            context_capacity: usize::MAX,
        };
        let err = ctx.decode(&[]).unwrap_err();
        assert!(matches!(err, LlamaError::Decode(message) if message.contains("empty")));
    }

    #[test]
    fn token_position_overflow_is_rejected_before_ffi() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: i32::MAX,
            batch_capacity: 512,
            context_capacity: usize::MAX,
        };
        let err = ctx.decode(&[1]).unwrap_err();
        assert!(matches!(err, LlamaError::Decode(message) if message.contains("overflow")));
    }
    #[test]
    fn chunk_partition_never_exceeds_configured_batch() {
        let sample: Vec<i32> = (0..1025).collect();
        let sizes: Vec<usize> = sample.chunks(256).map(|chunk| chunk.len()).collect();
        assert_eq!(sizes, vec![256, 256, 256, 256, 1]);
        let concatenated: Vec<i32> = sample.chunks(256).flat_map(|chunk| chunk.iter().copied()).collect();
        assert_eq!(concatenated, sample);
    }

    #[test]
    fn zero_capacity_rejected_before_native_call() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 0,
            batch_capacity: 0,
            context_capacity: usize::MAX,
        };
        let err = ctx.decode(&[1]).unwrap_err();
        assert!(matches!(err, LlamaError::Decode(message) if message.contains("capacity")));
    }

    /// Hardware/model acceptance test. CI cannot infer real GGUF parity from
    /// synthetic token chunks. Run explicitly on Tier 0 with a pinned model.
    #[test]
    #[ignore = "requires KILN_TEST_MODEL_PATH and real model inference"]
    fn real_model_chunked_prefill_matches_large_batch_top_token() {
        let model_path = std::env::var("KILN_TEST_MODEL_PATH")
            .expect("set KILN_TEST_MODEL_PATH to a pinned GGUF model file");
        let path = std::path::Path::new(&model_path);
        let threads = 4;
        let mut chunked = LlamaContext::load_with_ubatch(
            path, Some((2048, 128, 128, threads)),
        ).expect("load chunked context");
        let mut larger = LlamaContext::load_with_ubatch(
            path, Some((2048, 512, 128, threads)),
        ).expect("load larger-batch context");
        let prompt = "A reproducible test of numerical behavior across prompt batch boundaries. ".repeat(24);
        let tokens = chunked.tokenize(&prompt).expect("tokenize");
        assert!(tokens.len() > 128 && tokens.len() <= 512,
            "expected 129..=512 tokens, got {}", tokens.len());
        assert_eq!(tokens, larger.tokenize(&prompt).expect("second tokenize"));
        let chunked_logits = chunked.forward(&tokens).expect("chunked forward");
        let larger_logits = larger.forward(&tokens).expect("larger forward");
        assert_eq!(chunked_logits.len(), larger_logits.len());
        assert!(!chunked_logits.is_empty());
        fn argmax(xs: &[f32]) -> usize {
            xs.iter().enumerate()
                .filter(|(_, x)| x.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("finite logit")
                .0
        }
        assert_eq!(argmax(&chunked_logits), argmax(&larger_logits),
            "native chunking changed most likely next token");
        let max_error = chunked_logits.iter().zip(larger_logits.iter())
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error <= 0.1, "logits diverged; maximum absolute delta {max_error}");
    }

    /// Target-hardware Phase 3.5 A/B: replay the same executed token stream,
    /// with and without reuse of the already-materialized KV prefix.
    #[test]
    #[ignore = "requires a pinned KILN_TEST_MODEL_PATH GGUF on Tier 0"]
    fn append_only_kv_matches_full_prefix_logits() {
        use std::time::Instant;

        let model_path = std::env::var("KILN_TEST_MODEL_PATH")
            .expect("set KILN_TEST_MODEL_PATH to a pinned GGUF file");
        let path = std::path::Path::new(&model_path);
        let options = Some((2048, 256, 128, 4));
        let mut incremental = LlamaContext::load_with_ubatch(path, options)
            .expect("load incremental context");
        let mut full = LlamaContext::load_with_ubatch(path, options)
            .expect("load full-prefix context");

        // A nontrivial first-turn prefix makes the actual amount of prefill
        // avoided visible. Deliberately identical token IDs for each mode:
        // this is an inference-cache A/B, not a tokenizer-template test.
        let first_text = "User: Explain the phases of a simple physics experiment. ".repeat(16);
        let first = incremental.tokenize(&first_text).expect("tokenize first");
        let assistant = incremental.tokenize(" The steps are preparation, measurement, and analysis.")
            .expect("tokenize deterministic assistant continuation");
        let second = incremental.tokenize("\nUser: What is the next step?\nAssistant:")
            .expect("tokenize second turn");
        assert!(!first.is_empty() && !assistant.is_empty() && !second.is_empty());
        let mut entire = first.clone();
        entire.extend_from_slice(&assistant);
        entire.extend_from_slice(&second);
        assert!(entire.len() < 2048, "prefix requires {} tokens", entire.len());

        fn argmax(xs: &[f32]) -> usize {
            xs.iter().enumerate()
                .filter(|(_, value)| value.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("at least one finite model logit")
                .0
        }

        let repetitions = 5;
        let mut reused_ms = Vec::with_capacity(repetitions);
        let mut replayed_ms = Vec::with_capacity(repetitions);
        for trial in 0..repetitions {
            incremental.reset().expect("reset reusable context");
            full.reset().expect("reset full-prefix context");
            incremental.forward(&first).expect("prefill first turn");
            incremental.forward(&assistant).expect("prefill assistant response");

            // Alternate order of the measured operations to reduce a
            // systematic advantage from warmed CPU/memory state.
            let (append_logits, replay_logits) = if trial % 2 == 0 {
                let start = Instant::now();
                let append_logits = incremental.forward(&second).expect("incremental prefill");
                reused_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                let replay_logits = full.forward(&entire).expect("full-prefix replay");
                replayed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                (append_logits, replay_logits)
            } else {
                let start = Instant::now();
                let replay_logits = full.forward(&entire).expect("full-prefix replay");
                replayed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                let append_logits = incremental.forward(&second).expect("incremental prefill");
                reused_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                (append_logits, replay_logits)
            };
            assert_eq!(incremental.n_past(), full.n_past());
            assert_eq!(append_logits.len(), replay_logits.len());
            assert_eq!(argmax(&append_logits), argmax(&replay_logits),
                "KV reuse changed the greedy next token on trial {trial}");
            let max_diff = append_logits.iter().zip(replay_logits.iter())
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_diff <= 0.1,
                "KV reuse logits diverged by {max_diff} on trial {trial}");
        }
        reused_ms.sort_by(f64::total_cmp);
        replayed_ms.sort_by(f64::total_cmp);
        eprintln!(
            "phase35_kv_ab: trials={} prefix_tokens={} suffix_tokens={}              append_median_ms={:.3} replay_median_ms={:.3}              (observational only; not a Tier 0 speed acceptance)",
            repetitions, first.len() + assistant.len(), second.len(),
            reused_ms[repetitions / 2], replayed_ms[repetitions / 2]
        );
    }

    #[test]
    fn context_limit_rejects_one_more_token_without_native_call() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 16,
            batch_capacity: 512,
            context_capacity: 16,
        };
        let error = ctx.decode(&[1]).unwrap_err();
        assert!(matches!(error, LlamaError::Decode(message) if message.contains("n_ctx")));
        assert_eq!(ctx.n_past(), 16);
    }

    #[test]
    fn multi_chunk_context_overflow_rejected_before_partial_advance() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 8,
            batch_capacity: 4,
            context_capacity: 13,
        };
        let error = ctx.decode(&[1,2,3,4,5,6]).unwrap_err();
        assert!(matches!(error, LlamaError::Decode(message) if message.contains("n_ctx")));
        assert_eq!(ctx.n_past(), 8);
    }

}
