//! Per-layer KV cache.
//!
//! See docs/kv-cache-design.md. Stores post-bias, post-RoPE K and V
//! for every layer at every position already processed. The cache is
//! grown by append; it is reset when the caller's token sequence does
//! not start with the cached tokens.

/// K and V for one layer.
#[derive(Debug, Clone)]
pub struct LayerKv {
    /// [cap, kv_dim]
    pub k: Vec<f32>,
    /// [cap, kv_dim]
    pub v: Vec<f32>,
}

impl LayerKv {
    pub fn with_capacity(cap: usize, kv_dim: usize) -> Self {
        Self {
            k: Vec::with_capacity(cap * kv_dim),
            v: Vec::with_capacity(cap * kv_dim),
        }
    }

    pub fn len(&self, kv_dim: usize) -> usize {
        if kv_dim == 0 { 0 } else { self.k.len() / kv_dim }
    }

    pub fn push(&mut self, k_row: &[f32], v_row: &[f32]) {
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }
}

/// The whole cache. One LayerKv per transformer layer.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKv>,
    pub kv_dim: usize,
    pub max_len: usize,
    pub cached_tokens: Vec<u32>,
}

impl KvCache {
    pub fn new(num_layers: usize, kv_dim: usize, max_len: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerKv::with_capacity(max_len, kv_dim));
        }
        Self { layers, kv_dim, max_len, cached_tokens: Vec::with_capacity(max_len) }
    }

    pub fn len(&self) -> usize {
        self.cached_tokens.len()
    }

    pub fn clear(&mut self) {
        for l in self.layers.iter_mut() { l.clear(); }
        self.cached_tokens.clear();
    }

    pub fn can_append(&self, n: usize) -> bool {
        self.len() + n <= self.max_len
    }

    pub fn append_tokens(&mut self, tokens: &[u32]) {
        self.cached_tokens.extend_from_slice(tokens);
    }

    /// Read K row for layer L, position pos. Returns a slice of kv_dim.
    pub fn k_row(&self, layer: usize, pos: usize) -> &[f32] {
        let s = pos * self.kv_dim;
        &self.layers[layer].k[s..s + self.kv_dim]
    }

    pub fn v_row(&self, layer: usize, pos: usize) -> &[f32] {
        let s = pos * self.kv_dim;
        &self.layers[layer].v[s..s + self.kv_dim]
    }
}
