//! Convert a GGUF model to a different quantization. Phase 2: Q4_K
//! and Q6_K input, TQ1.0 output. See docs/convert-design.md.

use std::path::Path;

use crate::gguf::{GgufError, GgufFile, GgufTensorInfo, GgufType};
use crate::gguf_write::write_gguf;

fn ternarize(v: f32) -> f32 {
    if v < -0.5 { -1.0 } else if v > 0.5 { 1.0 } else { 0.0 }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut m = mant; let mut e = 0i32;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn dequant_tensor(bytes: &[u8], dtype: GgufType, numel: usize) -> Result<Vec<f32>, GgufError> {
    match dtype {
        GgufType::F32 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let o = i * 4;
                out.push(f32::from_le_bytes([bytes[o], bytes[o+1], bytes[o+2], bytes[o+3]]));
            }
            Ok(out)
        }
        GgufType::F16 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let o = i * 2;
                let h = u16::from_le_bytes([bytes[o], bytes[o+1]]);
                out.push(half_to_f32(h));
            }
            Ok(out)
        }
        GgufType::Q4_K => {
            let blocks = (numel + 255) / 256;
            let mut out = Vec::with_capacity(blocks * 256);
            for b in 0..blocks {
                let blk = &bytes[b * 144..b * 144 + 144];
                let vals = kiln_kernels::q4k_dequant_block(blk);
                out.extend_from_slice(&vals);
            }
            out.truncate(numel);
            Ok(out)
        }
        GgufType::Q6_K => {
            let blocks = (numel + 255) / 256;
            let mut out = Vec::with_capacity(blocks * 256);
            for b in 0..blocks {
                let blk = &bytes[b * 210..b * 210 + 210];
                let vals = kiln_kernels::q6k_dequant_block(blk);
                out.extend_from_slice(&vals);
            }
            out.truncate(numel);
            Ok(out)
        }
        other => Err(GgufError::Tensor(format!(
            "convert: unsupported input dtype {:?}", other))),
    }
}

fn pack_ternary(trits: &[f32]) -> Vec<u8> {
    kiln_kernels::pack(trits)
}

pub fn convert_to_tq1_0(input: &Path, output: &Path) -> Result<u64, GgufError> {
    let g = GgufFile::open(input)?;
    let num_tensors = g.tensors.len();

    let mut metadata: Vec<(String, crate::gguf::GgufValue)> = g.metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out_tensors: Vec<(GgufTensorInfo, Vec<u8>)> = Vec::with_capacity(num_tensors);
    for t in &g.tensors {
        let numel = t.num_elements() as usize;
        let src_bytes = g.tensor_bytes(&t.name)
            .ok_or_else(|| GgufError::Tensor(format!("missing bytes for {}", t.name)))?;
        // Only ternarize the per-layer weight tensors. Embeddings and
        // the output projection stay in their original precision: they
        // are read once per token, and their accuracy matters more
        // than their speed. This matches BitNet practice.
        // Only ternarize matmul weight tensors inside transformer
        // blocks. Biases are used in fp32 addition, not matmul.
        // Norms are read once per layer. Embeddings and the output
        // projection are read once per token. None of those are
        // ternarized.
        let ternarize_this = t.name.starts_with("blk.")
            && t.name.ends_with(".weight")
            && (t.name.contains(".attn_q.")
                || t.name.contains(".attn_k.")
                || t.name.contains(".attn_v.")
                || t.name.contains(".attn_output.")
                || t.name.contains(".ffn_gate.")
                || t.name.contains(".ffn_up.")
                || t.name.contains(".ffn_down."));
        if ternarize_this {
            let f32s = dequant_tensor(src_bytes, t.dtype, numel)?;
            let trits: Vec<f32> = f32s.iter().map(|&v| ternarize(v)).collect();
            let packed = pack_ternary(&trits);
            let info = GgufTensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: GgufType::Tq1_0,
                offset: 0,
            };
            out_tensors.push((info, packed));
        } else {
            // Pass through unchanged.
            let info = GgufTensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                offset: 0,
            };
            out_tensors.push((info, src_bytes.to_vec()));
        }
    }

    write_gguf(output, &metadata, &out_tensors)?;
    Ok(num_tensors as u64)
}
