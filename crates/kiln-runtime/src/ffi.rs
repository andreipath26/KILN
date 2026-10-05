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

extern "C" {
    fn kiln_llama_load(path: *const c_char) -> *mut c_void;
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
}

unsafe impl Send for LlamaContext {}

impl LlamaContext {
    pub fn load(path: &Path) -> Result<Self, LlamaError> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| LlamaError::Load(e.to_string()))?;
        let handle = unsafe { kiln_llama_load(cpath.as_ptr()) };
        if handle.is_null() {
            return Err(LlamaError::Load(format!(
                "shim returned null for {}",
                path.display()
            )));
        }
        let n_vocab = unsafe { kiln_llama_n_vocab(handle) };
        Ok(Self { handle, n_vocab, n_past: 0 })
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
        let mut buf: Vec<i32> = vec![0; text.len() + 8];
        let n = unsafe {
            kiln_llama_tokenize(
                self.handle,
                ctext.as_ptr(),
                text.len() as c_int,
                buf.as_mut_ptr(),
                buf.len() as c_int,
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
        let rc = unsafe {
            kiln_llama_decode(
                self.handle,
                tokens.as_ptr(),
                tokens.len() as c_int,
                self.n_past,
            )
        };
        if rc != 0 {
            return Err(LlamaError::Decode(format!("shim returned {}", rc)));
        }
        self.n_past += tokens.len() as i32;
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

    pub fn forward(&mut self, tokens: &[i32]) -> Result<Vec<f32>, LlamaError> {
        self.decode(tokens)?;
        self.logits()
    }

    pub fn token_to_str(&self, token: i32) -> String {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            kiln_llama_token_to_str(
                self.handle,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
        if n <= 0 { return String::new(); }
        buf.truncate(n as usize);
        String::from_utf8_lossy(&buf).to_string()
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { kiln_llama_free(self.handle); }
    }
}
