//! The single backend for Phase 1.5: CPU scalar.
//! See docs/dispatch-integration-design.md.

use crate::dispatch::{Backend, DispatchError, ExecContext, Operation, QuantKind};

pub struct CpuScalarBackend;

impl CpuScalarBackend {
    pub fn new() -> Self { Self }
}

impl Default for CpuScalarBackend {
    fn default() -> Self { Self::new() }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else {
            let mut m = mant; let mut e = 0i32;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            let exp32 = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (exp32 << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn decode_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 4;
        if off + 4 > bytes.len() { break; }
        let b = &bytes[off..off + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

fn dot_f32(bytes: &[u8], n: usize, x: &[f32]) -> f32 {
    let w = decode_f32(bytes, n);
    let mut acc = 0.0f32;
    for i in 0..n { acc += w[i] * x[i]; }
    acc
}

fn dot_f16(bytes: &[u8], n: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n {
        let b = &bytes[i * 2..i * 2 + 2];
        let h = u16::from_le_bytes([b[0], b[1]]);
        acc += f16_to_f32(h) * x[i];
    }
    acc
}

impl Backend for CpuScalarBackend {
    fn name(&self) -> &str { "cpu_scalar" }

    fn supports(&self, op: &Operation) -> bool {
        match op {
            Operation::Matmul { quant, .. } => !matches!(quant, QuantKind::TQ1_0),
            _ => true,
        }
    }

    fn estimate(&self, _op: &Operation) -> Option<u64> { Some(0) }

    fn execute(&self, op: &Operation, ctx: &mut ExecContext)
        -> Result<(), DispatchError>
    {
        match op {
            Operation::Matmul { quant, in_features, .. } => {
                let w = ctx.weight_bytes;
                let n = *in_features;
                let result = match quant {
                    QuantKind::Q4K => kiln_kernels::q4k_matmul_scalar(w, n, ctx.activations),
                    QuantKind::Q6K => kiln_kernels::q6k_matmul_scalar(w, n, ctx.activations),
                    QuantKind::TQ1_0 => {
                        let acts_i8: Vec<i8> = ctx.activations.iter().map(|&v| v as i8).collect();
                        kiln_kernels::matmul_scalar(w, &acts_i8, n, 1.0)
                    }
                    QuantKind::F32 => dot_f32(w, n, ctx.activations),
                    QuantKind::F16 => dot_f16(w, n, ctx.activations),
                };
                if !ctx.out.is_empty() { ctx.out[0] = result; }
                Ok(())
            }
            Operation::RmsNorm { size } => {
                let n = *size;
                let mut sum_sq = 0.0f32;
                for i in 0..n { sum_sq += ctx.activations[i] * ctx.activations[i]; }
                let rms = (sum_sq / n as f32 + 1e-6).sqrt();
                for i in 0..n { ctx.out[i] = ctx.activations[i] / rms; }
                Ok(())
            }
            Operation::Embedding { .. } => Ok(()),
            Operation::Attention { .. } => Err(DispatchError::Backend(
                "attention is not dispatched in Phase 1.5".to_string())),
            Operation::Sample { .. } => Err(DispatchError::Backend(
                "sample is not dispatched in Phase 1.5".to_string())),
        }
    }
}
