//! Rust wrapper over the KILN llama.cpp shim.
//! The shim lives in cpp/llama_shim/. Rust never sees llama.cpp
//! structs — every call goes through a stable extern "C" ABI.
//! See docs/runtime-loading-design.md.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
const MAX_KV_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;

/// Opaque, model-context-local KV/logit snapshot. Not portable between
/// different loaded contexts, machines, or inference configurations.
#[derive(Debug)]
pub struct KvSnapshot {
    state: Vec<u8>,
    context_id: u64,
    n_past: i32,
}

impl KvSnapshot {
    pub fn token_positions(&self) -> usize { self.n_past as usize }
    pub fn bytes(&self) -> usize { self.state.len() }
}


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
    pub lazy_moe_experts: c_int,
}

extern "C" {
    fn kiln_llama_load(path: *const c_char, params: *const LlamaParamsRaw) -> *mut c_void;
    fn kiln_llama_free(h: *mut c_void);
    fn kiln_llama_n_vocab(h: *mut c_void) -> c_int;
    fn kiln_llama_mask_token(h: *mut c_void) -> c_int;
    fn kiln_llama_shift_logits(h: *mut c_void) -> c_int;
    fn kiln_llama_apply_user_template(
        h: *mut c_void, prompt: *const c_char, prompt_len: c_int,
        out: *mut c_char, out_cap: c_int,
    ) -> c_int;
    fn kiln_llama_diffusion_step(
        h: *mut c_void, tokens: *const c_int, n_tokens: c_int,
        predicted_tokens: *mut c_int, confidence: *mut f32,
        attention: *mut f32, attention_capacity: usize,
    ) -> c_int;
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
        request_last_logits: c_int,
    ) -> c_int;
    fn kiln_llama_logits(h: *mut c_void, out: *mut f32, out_cap: c_int) -> c_int;
    fn kiln_llama_greedy_token(h: *mut c_void) -> c_int;
    fn kiln_llama_verify_greedy_draft(
        h: *mut c_void, draft: *const c_int, n: c_int,
        next_token: *mut c_int,
    ) -> c_int;
    fn kiln_llama_token_to_str(
        h: *mut c_void,
        token: c_int,
        buf: *mut c_char,
        buf_len: c_int,
    ) -> c_int;
    fn kiln_llama_state_size(h: *mut c_void) -> usize;
    fn kiln_llama_state_get(h: *mut c_void, dst: *mut u8, capacity: usize) -> usize;
    fn kiln_llama_state_set(
        h: *mut c_void, src: *const u8, length: usize, n_past: c_int,
    ) -> c_int;
    fn kiln_llama_reset(h: *mut c_void) -> c_int;
    fn kiln_llama_eos_token(h: *mut c_void) -> c_int;
}

/// Interpret llama_token_to_piece's negative required-length return safely.
fn required_token_piece_capacity(native_length: i32, supplied: usize) -> Result<usize, LlamaError> {
    const MAX_TOKEN_PIECE_BYTES: usize = 1024 * 1024;
    let required = native_length.checked_neg()
        .and_then(|size| usize::try_from(size).ok())
        .ok_or_else(|| LlamaError::Decode("invalid native token piece length".into()))?;
    if native_length >= 0 || required <= supplied || required > MAX_TOKEN_PIECE_BYTES {
        return Err(LlamaError::Decode(format!(
            "native token piece length {required} is invalid or exceeds safety limit"
        )));
    }
    Ok(required)
}

pub struct LlamaContext {
    handle: *mut c_void,
    n_vocab: i32,
    n_past: i32,
    /// Physical reusable batch capacity configured in the C shim.
    batch_capacity: usize,
    /// Maximum token positions available in this unshifted context.
    context_capacity: usize,
    context_id: u64,
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
        Self::load_with_ubatch_and_lazy_moe(path, params, false)
    }

    /// Load with optional on-demand mmap reads for supported MoE expert tensors.
    pub fn load_with_ubatch_and_lazy_moe(
        path: &Path,
        params: Option<(i32, i32, i32, i32)>,
        lazy_moe_experts: bool,
    ) -> Result<Self, LlamaError> {
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
            lazy_moe_experts: if lazy_moe_experts { 1 } else { 0 },
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
        let context_id = NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self { handle, n_vocab, n_past: 0, batch_capacity, context_capacity, context_id })
    }

    pub fn n_vocab(&self) -> i32 { self.n_vocab }

    /// Return the model's diffusion mask token; fails for autoregressive models.
    pub fn mask_token(&self) -> Result<i32, LlamaError> {
        let token = unsafe { kiln_llama_mask_token(self.handle) };
        if token < 0 { Err(LlamaError::Decode("loaded model is not a supported diffusion model".into())) }
        else { Ok(token) }
    }

    /// Read the LLaDA position-to-logit mapping from model metadata.
    pub fn diffusion_shift_logits(&self) -> Result<bool, LlamaError> {
        match unsafe { kiln_llama_shift_logits(self.handle) } {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(LlamaError::Decode("loaded model has no diffusion logit-shift setting".into())),
        }
    }

    /// Format a one-turn user prompt with the model's native chat template.
    pub fn format_user_prompt(&self, prompt: &str) -> Result<String, LlamaError> {
        let cprompt = CString::new(prompt).map_err(|e| LlamaError::Tokenize(e.to_string()))?;
        let prompt_len = c_int::try_from(prompt.len())
            .map_err(|_| LlamaError::Tokenize("prompt exceeds the C ABI length limit".into()))?;
        let mut cap = prompt.len().checked_mul(2).and_then(|n| n.checked_add(512))
            .ok_or_else(|| LlamaError::Tokenize("formatted prompt length overflow".into()))?;
        cap = cap.max(1024);
        for _ in 0..3 {
            let native_cap = c_int::try_from(cap)
                .map_err(|_| LlamaError::Tokenize("formatted prompt exceeds C ABI capacity".into()))?;
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                kiln_llama_apply_user_template(
                    self.handle, cprompt.as_ptr(), prompt_len,
                    buf.as_mut_ptr() as *mut c_char, native_cap,
                )
            };
            if n < 0 {
                return Err(LlamaError::Tokenize(format!("model chat template failed ({n})")));
            }
            let written = n as usize;
            if written >= cap {
                cap = written.checked_add(1)
                    .ok_or_else(|| LlamaError::Tokenize("formatted prompt length overflow".into()))?;
                continue;
            }
            buf.truncate(written);
            return String::from_utf8(buf)
                .map_err(|e| LlamaError::Tokenize(format!("chat template produced invalid UTF-8: {e}")));
        }
        Err(LlamaError::Tokenize("chat template output exceeded retry bound".into()))
    }

    /// Run a full non-causal diffusion forward pass, returning top-1 token
    /// proposals, their probabilities, and the CLAD attention matrix.
    pub fn diffusion_step(
        &mut self, tokens: &[i32],
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<f32>), LlamaError> {
        if tokens.is_empty() || tokens.len() > self.batch_capacity || tokens.len() > self.context_capacity {
            return Err(LlamaError::Decode("diffusion sequence must fit configured context and batch".into()));
        }
        let n = c_int::try_from(tokens.len())
            .map_err(|_| LlamaError::Decode("diffusion sequence exceeds C ABI length".into()))?;
        let attention_capacity = tokens.len().checked_mul(tokens.len())
            .ok_or_else(|| LlamaError::Decode("attention matrix size overflow".into()))?;
        let mut predicted = vec![0i32; tokens.len()];
        let mut confidence = vec![0f32; tokens.len()];
        let mut attention = vec![0f32; attention_capacity];
        let rc = unsafe {
            kiln_llama_diffusion_step(
                self.handle, tokens.as_ptr(), n,
                predicted.as_mut_ptr(), confidence.as_mut_ptr(),
                attention.as_mut_ptr(), attention.len(),
            )
        };
        if rc != 0 {
            self.n_past = 0;
            return Err(LlamaError::Decode(format!("native diffusion forward returned {rc}")));
        }
        self.n_past = 0;
        Ok((predicted, confidence, attention))
    }
    pub fn n_past(&self) -> i32 { self.n_past }

    /// Take an explicit, bounded in-memory snapshot of the model's KV,
    /// logits and recurrent state. This does NOT write a session file and
    /// intentionally cannot be restored into a different context.
    pub fn snapshot_kv(&self, requested_max_bytes: usize) -> Result<KvSnapshot, LlamaError> {
        if self.handle.is_null() || self.n_past <= 0 {
            return Err(LlamaError::Decode("KV snapshot requires an active decoded session".into()));
        }
        if requested_max_bytes == 0 || requested_max_bytes > MAX_KV_SNAPSHOT_BYTES {
            return Err(LlamaError::Decode("KV snapshot byte cap must fit 1..=256 MiB".into()));
        }
        let size = unsafe { kiln_llama_state_size(self.handle) };
        if size == 0 || size > requested_max_bytes {
            return Err(LlamaError::Decode("KV state exceeds allowed snapshot memory".into()));
        }
        let mut state = vec![0u8; size];
        let written = unsafe {
            kiln_llama_state_get(self.handle, state.as_mut_ptr(), state.len())
        };
        if written == 0 || written > size {
            return Err(LlamaError::Decode("native KV snapshot returned invalid length".into()));
        }
        state.truncate(written);
        Ok(KvSnapshot { state, context_id: self.context_id, n_past: self.n_past })
    }

    /// Restore an opaque snapshot to exactly the context that produced it.
    /// On native restoration failure, discard the old cache state and require
    /// caller re-prefill instead of continuing with inconsistent positions.
    pub fn restore_kv(&mut self, snapshot: &KvSnapshot) -> Result<(), LlamaError> {
        if self.handle.is_null() || snapshot.context_id != self.context_id ||
           snapshot.n_past <= 0 || snapshot.n_past as usize > self.context_capacity ||
           snapshot.state.is_empty() || snapshot.state.len() > MAX_KV_SNAPSHOT_BYTES {
            return Err(LlamaError::Decode("KV snapshot is incompatible with this context".into()));
        }
        let result = unsafe { kiln_llama_state_set(
            self.handle, snapshot.state.as_ptr(), snapshot.state.len(), snapshot.n_past,
        ) };
        if result != 0 {
            self.reset().map_err(|e| LlamaError::Decode(format!(
                "KV restoration failed and reset failed: {e}")))?;
            return Err(LlamaError::Decode("KV restoration failed; session reset".into()));
        }
        self.n_past = snapshot.n_past;
        Ok(())
    }

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
        let chunk_count = tokens.len().div_ceil(self.batch_capacity);
        for (index, chunk) in tokens.chunks(self.batch_capacity).enumerate() {
            let count = i32::try_from(chunk.len())
                .map_err(|_| LlamaError::Decode("token batch exceeds the C ABI limit".to_string()))?;
            let rc = unsafe {
                kiln_llama_decode(
                    self.handle,
                    chunk.as_ptr(),
                    count,
                    self.n_past,
                    i32::from(index + 1 == chunk_count),
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

    /// Verify a short batch of draft tokens against the target model.
    /// Only the accepted prefix becomes resident in KV. The returned token
    /// is the next target-model greedy token, NOT YET evaluated into KV.
    /// On any native error, reset the session so callers cannot accidentally
    /// continue after a partially modified KV state.
    pub fn verify_greedy_draft(&mut self, draft: &[i32]) -> Result<(usize, i32), LlamaError> {
        if draft.is_empty() || draft.len() > self.batch_capacity || draft.len() > 16 {
            return Err(LlamaError::Decode("draft length must fit 1..=min(16,n_batch)".into()));
        }
        let len = i32::try_from(draft.len()).map_err(|_| LlamaError::Decode("draft too long".into()))?;
        let end = self.n_past.checked_add(len)
            .ok_or_else(|| LlamaError::Decode("draft position overflow".into()))?;
        if end as usize > self.context_capacity {
            return Err(LlamaError::Decode("draft exceeds configured context".into()));
        }
        let mut next = -1i32;
        let accepted = unsafe {
            kiln_llama_verify_greedy_draft(self.handle, draft.as_ptr(), len, &mut next)
        };
        if accepted < 0 || accepted > len || next < 0 {
            // Keep fail-safe native/Rust position synchronization.
            self.reset().map_err(|e| LlamaError::Decode(format!(
                "draft verification failed; reset also failed: {e}")))?;
            return Err(LlamaError::Decode("native draft verification failed; session reset".into()));
        }
        self.n_past += accepted;
        Ok((accepted as usize, next))
    }

    pub fn forward(&mut self, tokens: &[i32]) -> Result<Vec<f32>, LlamaError> {
        self.decode(tokens)?;
        self.logits()
    }

    /// Decode one token to raw UTF-8 bytes. Do not lossy-convert here;
    /// a token may carry a partial multi-byte sequence that the next
    /// token completes. The caller buffers bytes and decodes when a
    /// full sequence is available.
    pub fn token_bytes(&self, token: i32) -> Result<Vec<u8>, LlamaError> {
        const INITIAL_BYTES: usize = 256;
        let mut buf = vec![0u8; INITIAL_BYTES];
        // llama_token_to_piece returns a negative *required length* when the
        // supplied buffer is too small; the original code silently returned
        // empty bytes, corrupting output for larger token pieces.
        for _attempt in 0..2 {
            let n = unsafe {
                kiln_llama_token_to_str(
                    self.handle,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    c_int::try_from(buf.len()).map_err(|_| {
                        LlamaError::Decode("token piece buffer exceeds native capacity".into())
                    })?,
                )
            };
            if n >= 0 {
                let size = usize::try_from(n).map_err(|_| {
                    LlamaError::Decode("invalid token piece length".into())
                })?;
                if size > buf.len() {
                    return Err(LlamaError::Decode("token piece exceeded allocated buffer".into()));
                }
                buf.truncate(size);
                return Ok(buf);
            }
            let required = required_token_piece_capacity(n, buf.len())?;
            buf.resize(required, 0);
        }
        Err(LlamaError::Decode("native token piece size changed during retry".into()))
    }

    /// Backwards-compatible string form. Lossy. Prefer token_bytes.
    pub fn token_to_str(&self, token: i32) -> String {
        self.token_bytes(token)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default()
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
    fn kv_snapshot_rejects_invalid_state_without_native_calls() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(), n_vocab: 10, n_past: 7,
            batch_capacity: 8, context_capacity: 16, context_id: 99,
        };
        assert!(ctx.snapshot_kv(1024).is_err());
        let foreign = KvSnapshot { state: vec![0;16], context_id: 100, n_past: 7 };
        assert!(ctx.restore_kv(&foreign).is_err());
        let oversized_pos = KvSnapshot { state: vec![0;16], context_id: 99, n_past: 17 };
        assert!(ctx.restore_kv(&oversized_pos).is_err());
        let empty = KvSnapshot { state: vec![], context_id: 99, n_past: 7 };
        assert!(ctx.restore_kv(&empty).is_err());
        assert_eq!(ctx.n_past(), 7);
        assert_eq!(foreign.token_positions(), 7);
        assert_eq!(foreign.bytes(), 16);
    }

    #[test]
    #[ignore = "requires pinned real GGUF and native KV continuation"]
    fn native_kv_snapshot_restores_identical_next_greedy_token() {
        let model = std::env::var("KILN_TEST_MODEL_PATH")
            .expect("set KILN_TEST_MODEL_PATH to a pinned local GGUF");
        let mut ctx = LlamaContext::load_with_ubatch(
            Path::new(&model), Some((2048,256,128,2))
        ).unwrap();
        let prompt = ctx.tokenize("Local KV snapshot continuity is verified.").unwrap();
        let first = ctx.forward_greedy(&prompt).unwrap();
        let snapshot = ctx.snapshot_kv(256*1024*1024).unwrap();
        let before = ctx.forward_greedy(&[first]).unwrap();
        ctx.restore_kv(&snapshot).unwrap();
        assert_eq!(ctx.n_past() as usize, snapshot.token_positions());
        let after = ctx.forward_greedy(&[first]).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn empty_decode_batch_is_rejected_before_ffi() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 0,
            batch_capacity: 512,
            context_capacity: usize::MAX,
            context_id: 0,
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
            context_id: 0,
        };
        let err = ctx.decode(&[1]).unwrap_err();
        assert!(matches!(err, LlamaError::Decode(message) if message.contains("overflow")));
    }
    #[test]
    fn only_final_prompt_chunk_needs_logits() {
        let tokens: Vec<i32> = (0..1025).collect();
        let count = tokens.len().div_ceil(256);
        let flags: Vec<_> = tokens.chunks(256).enumerate()
            .map(|(index, _)| index + 1 == count).collect();
        assert_eq!(flags, vec![false, false, false, false, true]);
        let single: Vec<_> = tokens[..10].chunks(256).enumerate()
            .map(|(index, _)| index + 1 == 1).collect();
        assert_eq!(single, vec![true]);
    }

    /// Real-model verification of target-greedy correction and KV tail trim.
    #[test]
    #[ignore = "requires pinned KILN_TEST_MODEL_PATH GGUF"]
    fn real_model_draft_rejection_restores_identical_kv_continuation() {
        let file = std::env::var("KILN_TEST_MODEL_PATH")
            .expect("set KILN_TEST_MODEL_PATH to a pinned local GGUF");
        let path = std::path::Path::new(&file);
        let params = Some((2048, 256, 128, 4));
        let mut baseline = LlamaContext::load_with_ubatch(path, params).unwrap();
        let mut spec = LlamaContext::load_with_ubatch(path, params).unwrap();
        let tokens = baseline.tokenize(
            "Explain why we measure numerical outputs after a cache rollback."
        ).unwrap();
        assert_eq!(tokens, spec.tokenize(
            "Explain why we measure numerical outputs after a cache rollback."
        ).unwrap());
        let first = baseline.forward_greedy(&tokens).unwrap();
        assert_eq!(first, spec.forward_greedy(&tokens).unwrap());
        let second = baseline.forward_greedy(&[first]).unwrap();
        let wrong = (second + 1) % spec.n_vocab();
        let (accepted, corrected) = spec.verify_greedy_draft(&[first, wrong]).unwrap();
        assert_eq!(accepted, 1);
        assert_eq!(corrected, second);
        assert_eq!(spec.n_past(), baseline.n_past());
        let base_next = baseline.forward_greedy(&[second]).unwrap();
        let spec_next = spec.forward_greedy(&[corrected]).unwrap();
        assert_eq!(base_next, spec_next);
    }

    #[test]
    fn draft_validation_happens_before_native_call() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(), n_vocab: 64, n_past: 10,
            batch_capacity: 4, context_capacity: 12, context_id: 0,
        };
        assert!(ctx.verify_greedy_draft(&[]).is_err());
        assert!(ctx.verify_greedy_draft(&[1, 2, 3, 4, 5]).is_err());
        assert!(ctx.verify_greedy_draft(&[1, 2, 3]).is_err());
        assert_eq!(ctx.n_past(), 10);
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
            context_id: 0,
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

    /// Validate the native fast path against full-logits Rust greedy sampling
    /// on the same pinned GGUF. No model weights are available in hosted CI.
    #[test]
    #[ignore = "requires KILN_TEST_MODEL_PATH to a pinned GGUF"]
    fn native_greedy_matches_full_logits_for_real_model() {
        use kiln_core::chat::{Sampler, SamplingStrategy};
        let filename = std::env::var("KILN_TEST_MODEL_PATH")
            .expect("set KILN_TEST_MODEL_PATH to a pinned GGUF model");
        let path = std::path::Path::new(&filename);
        let params = Some((2048, 256, 128, 4));
        let mut native = LlamaContext::load_with_ubatch(path, params)
            .expect("native session");
        let mut full = LlamaContext::load_with_ubatch(path, params)
            .expect("full logits session");
        let text = "An apple falls from a tree because";
        let ids = full.tokenize(text).expect("tokenize prompt");
        assert_eq!(ids, native.tokenize(text).expect("tokenize prompt twice"));
        assert!(!ids.is_empty());

        let mut rust_sampler = Sampler::new(SamplingStrategy::Greedy, 42);
        let mut rust_token = rust_sampler.sample(&full.forward(&ids).expect("prefill"))
            .expect("nonempty logits") as i32;
        let mut native_token = native.forward_greedy(&ids).expect("native prefill");
        assert_eq!(rust_token, native_token, "native greedy prefill mismatch");

        for step in 0..32 {
            let next_rust = full.forward(&[rust_token]).expect("decode full logits");
            let next_native = native.forward_greedy(&[native_token]).expect("decode native");
            rust_token = rust_sampler.sample(&next_rust).expect("sample next") as i32;
            native_token = next_native;
            assert_eq!(rust_token, native_token, "greedy mismatch on step {step}");
        }
    }

    #[test]
    fn context_limit_rejects_one_more_token_without_native_call() {
        let mut ctx = LlamaContext {
            handle: std::ptr::null_mut(),
            n_vocab: 1,
            n_past: 16,
            batch_capacity: 512,
            context_capacity: 16,
            context_id: 0,
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
            context_id: 0,
        };
        let error = ctx.decode(&[1,2,3,4,5,6]).unwrap_err();
        assert!(matches!(error, LlamaError::Decode(message) if message.contains("n_ctx")));
        assert_eq!(ctx.n_past(), 8);
    }

    #[test]
    fn token_piece_capacity_retries_only_valid_native_required_lengths() {
        assert_eq!(required_token_piece_capacity(-400, 256).unwrap(), 400);
        assert!(required_token_piece_capacity(-255, 256).is_err());
        assert!(required_token_piece_capacity(-1_048_577, 256).is_err());
        assert!(required_token_piece_capacity(i32::MIN, 256).is_err());
        assert!(required_token_piece_capacity(0, 256).is_err());
        assert!(required_token_piece_capacity(400, 256).is_err());
    }

}
