//! Phase 2 backend: TQ1.0 ternary matmul.
//! Supports only Matmul with QuantKind::TQ1_0.
//! See docs/ternary-runtime-design.md.

use crate::dispatch::{Backend, DispatchError, ExecContext, Operation, QuantKind};

pub struct CpuTernaryBackend;

impl CpuTernaryBackend {
    pub fn new() -> Self { Self }
}

impl Default for CpuTernaryBackend {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuTernaryBackend {
    fn name(&self) -> &str { "cpu_ternary_tq1_0" }

    fn supports(&self, op: &Operation) -> bool {
        matches!(op, Operation::Matmul { quant: QuantKind::TQ1_0, .. })
    }

    fn estimate(&self, _op: &Operation) -> Option<u64> { Some(0) }

    fn execute(&self, op: &Operation, ctx: &mut ExecContext)
        -> Result<(), DispatchError>
    {
        match op {
            Operation::Matmul { in_features, .. } => {
                let n = *in_features;
                // Activation quantization. The TQ1.0 kernel computes an
                // integer dot product. To preserve the activation
                // magnitudes, scale the vector so its largest absolute
                // value maps to 127, cast to i8, run the kernel, then
                // rescale the result.
                let mut max_abs = 0.0f32;
                for i in 0..n {
                    let v = ctx.activations[i].abs();
                    if v > max_abs { max_abs = v; }
                }
                let a_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
                let inv_scale = if a_scale > 0.0 { 1.0 / a_scale } else { 1.0 };
                let acts_i8: Vec<i8> = ctx.activations[..n].iter()
                    .map(|&v| {
                        let q = v * a_scale;
                        let c = q.round().clamp(-127.0, 127.0);
                        c as i8
                    })
                    .collect();
                let raw = kiln_kernels::matmul_scalar(ctx.weight_bytes, &acts_i8, n, 1.0);
                let r = raw * inv_scale;
                if !ctx.out.is_empty() { ctx.out[0] = r; }
                Ok(())
            }
            _ => Err(DispatchError::Backend(
                "cpu_ternary only handles Matmul TQ1_0".to_string())),
        }
    }
}
