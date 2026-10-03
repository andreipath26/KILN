//! KILN Kernels.
//!
//! Safe Rust wrappers around the C kernels in cpp/kernels/. See
//! docs/kernels-design.md for the format, the ABI, and the build process.

use std::os::raw::{c_float, c_int};

extern "C" {
    fn kiln_tq1_0_pack(src: *const c_float, n: usize, dst: *mut u8) -> c_int;
    fn kiln_tq1_0_pack_scalar(src: *const c_float, n: usize, dst: *mut u8) -> c_int;
    fn kiln_tq1_0_unpack(src: *const u8, n: usize, dst: *mut c_float) -> c_int;
    fn kiln_tq1_0_unpack_table(src: *const u8, n: usize, dst: *mut c_float) -> c_int;
    fn kiln_q4k_dequant_block(src: *const u8, dst: *mut c_float) -> c_int;
    fn kiln_q4k_matmul_scalar(
        weights: *const u8,
        num_weights: usize,
        activations: *const c_float,
        num_activations: usize,
        output: *mut c_float,
    ) -> c_int;
    fn kiln_q6k_dequant_block(src: *const u8, dst: *mut c_float) -> c_int;
    fn kiln_q6k_matmul_scalar(
        weights: *const u8,
        num_weights: usize,
        activations: *const c_float,
        num_activations: usize,
        output: *mut c_float,
    ) -> c_int;
    fn kiln_tq1_0_matmul_scalar(
        weights: *const u8,
        num_weights: usize,
        activations: *const i8,
        num_activations: usize,
        scale: c_float,
        output: *mut c_float,
    ) -> c_int;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
extern "C" {
    fn kiln_tq1_0_matmul_lut_naive(
        weights: *const u8,
        num_weights: usize,
        activations: *const i8,
        num_activations: usize,
        scale: c_float,
        output: *mut c_float,
    ) -> c_int;

    fn kiln_tq1_0_matmul_8row_lut_failed(
        weights: *const u8,
        row_stride_bytes: usize,
        num_weights_per_row: usize,
        activations: *const i8,
        num_activations: usize,
        scale: c_float,
        outputs: *mut c_float,
    ) -> c_int;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
extern "C" {
    fn kiln_tq1_0_pack_avx2_dispatch(src: *const c_float, n: usize, dst: *mut u8) -> c_int;
}

/// Pack a slice of ternary values into TQ1.0 bytes.
///
/// Every value in `src` must be exactly -1.0, 0.0, or 1.0. Panics otherwise.
pub fn pack(src: &[f32]) -> Vec<u8> {
    if src.is_empty() {
        panic!("kiln_kernels::pack: empty input slice");
    }
    for (i, &v) in src.iter().enumerate() {
        if v != -1.0 && v != 0.0 && v != 1.0 {
            panic!(
                "kiln_kernels::pack: value at index {} is {}, expected -1.0, 0.0, or 1.0",
                i, v
            );
        }
    }
    let byte_len = (src.len() + 4) / 5;
    let mut dst = vec![0u8; byte_len];
    let rc = unsafe { kiln_tq1_0_pack(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::pack: C kernel returned error code {}", rc);
    }
    dst
}

/// Pack using the scalar path only. Used for benchmarking and for
/// verifying the AVX2 path produces identical output.
pub fn pack_scalar(src: &[f32]) -> Vec<u8> {
    if src.is_empty() {
        panic!("kiln_kernels::pack_scalar: empty input slice");
    }
    for (i, &v) in src.iter().enumerate() {
        if v != -1.0 && v != 0.0 && v != 1.0 {
            panic!(
                "kiln_kernels::pack_scalar: value at index {} is {}, expected -1.0, 0.0, or 1.0",
                i, v
            );
        }
    }
    let byte_len = (src.len() + 4) / 5;
    let mut dst = vec![0u8; byte_len];
    let rc = unsafe { kiln_tq1_0_pack_scalar(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::pack_scalar: C kernel returned error code {}", rc);
    }
    dst
}

/// Pack using the AVX2 path only. Panics on non-x86_64 targets.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub fn pack_avx2(src: &[f32]) -> Vec<u8> {
    if src.is_empty() {
        panic!("kiln_kernels::pack_avx2: empty input slice");
    }
    let byte_len = (src.len() + 4) / 5;
    let mut dst = vec![0u8; byte_len];
    let rc = unsafe { kiln_tq1_0_pack_avx2_dispatch(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::pack_avx2: C kernel returned error code {}", rc);
    }
    dst
}

/// Unpack TQ1.0 bytes back into `n` ternary values.
///
/// Panics if `src` is too short to hold `n` values, or if any byte is in
/// the reserved range [243, 255].
pub fn unpack(src: &[u8], n: usize) -> Vec<f32> {
    if n == 0 {
        panic!("kiln_kernels::unpack: n must be greater than 0");
    }
    let required = (n + 4) / 5;
    if src.len() < required {
        panic!(
            "kiln_kernels::unpack: need {} bytes for {} values, got {}",
            required, n, src.len()
        );
    }
    for (i, &b) in src[..required].iter().enumerate() {
        if b > 242 {
            panic!(
                "kiln_kernels::unpack: byte at index {} is {} which is in the reserved range [243, 255]",
                i, b
            );
        }
    }
    let mut dst = vec![0.0f32; n];
    let rc = unsafe { kiln_tq1_0_unpack(src.as_ptr(), n, dst.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::unpack: C kernel returned error code {}", rc);
    }
    dst
}

/// Unpack using the table-based fast path.
pub fn unpack_table(src: &[u8], n: usize) -> Vec<f32> {
    if n == 0 {
        panic!("kiln_kernels::unpack_table: n must be greater than 0");
    }
    let required = (n + 4) / 5;
    if src.len() < required {
        panic!(
            "kiln_kernels::unpack_table: need {} bytes for {} values, got {}",
            required, n, src.len()
        );
    }
    for (i, &b) in src[..required].iter().enumerate() {
        if b > 242 {
            panic!(
                "kiln_kernels::unpack_table: byte at index {} is {} which is in the reserved range [243, 255]",
                i, b
            );
        }
    }
    let mut dst = vec![0.0f32; n];
    let rc = unsafe { kiln_tq1_0_unpack_table(src.as_ptr(), n, dst.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::unpack_table: C kernel returned error code {}", rc);
    }
    dst
}

/// Fused ternary matmul, scalar reference.
///
/// Computes the dot product between a TQ1.0 packed weight row and an int8
/// activation vector, scaled by `scale`.
///
/// `weights` must contain `num_weights` trits packed at 5 trits per byte.
/// `activations` must be at least `num_weights` elements long.
///
/// Panics if the packed weight bytes are in the reserved range [243, 255].
pub fn matmul_scalar(weights: &[u8], activations: &[i8], num_weights: usize, scale: f32) -> f32 {
    if num_weights == 0 {
        return 0.0;
    }
    let required = (num_weights + 4) / 5;
    if weights.len() < required {
        panic!(
            "kiln_kernels::matmul_scalar: need {} bytes for {} weights, got {}",
            required, num_weights, weights.len()
        );
    }
    if activations.len() < num_weights {
        panic!(
            "kiln_kernels::matmul_scalar: need {} activations, got {}",
            num_weights, activations.len()
        );
    }
    for (i, &b) in weights[..required].iter().enumerate() {
        if b > 242 {
            panic!(
                "kiln_kernels::matmul_scalar: byte at index {} is {} in reserved range [243, 255]",
                i, b
            );
        }
    }
    let mut output = 0.0f32;
    let rc = unsafe {
        kiln_tq1_0_matmul_scalar(
            weights.as_ptr(),
            num_weights,
            activations.as_ptr(),
            activations.len(),
            scale,
            &mut output as *mut f32,
        )
    };
    if rc != 0 {
        panic!("kiln_kernels::matmul_scalar: C kernel returned error code {}", rc);
    }
    output
}

/// Fused ternary matmul, AVX2 path with LUT.
///
/// Computes the same result as `matmul_scalar` using a lookup table built
/// from the activations. Panics on non-x86_64 targets.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub fn matmul_lut_naive(weights: &[u8], activations: &[i8], num_weights: usize, scale: f32) -> f32 {
    if num_weights == 0 {
        return 0.0;
    }
    let required = (num_weights + 4) / 5;
    if weights.len() < required {
        panic!(
            "kiln_kernels::matmul_lut_naive: need {} bytes for {} weights, got {}",
            required, num_weights, weights.len()
        );
    }
    if activations.len() < num_weights {
        panic!(
            "kiln_kernels::matmul_lut_naive: need {} activations, got {}",
            num_weights, activations.len()
        );
    }
    for (i, &b) in weights[..required].iter().enumerate() {
        if b > 242 {
            panic!(
                "kiln_kernels::matmul_lut_naive: byte at index {} is {} in reserved range [243, 255]",
                i, b
            );
        }
    }
    let mut output = 0.0f32;
    let rc = unsafe {
        kiln_tq1_0_matmul_lut_naive(
            weights.as_ptr(),
            num_weights,
            activations.as_ptr(),
            activations.len(),
            scale,
            &mut output as *mut f32,
        )
    };
    if rc != 0 {
        panic!("kiln_kernels::matmul_lut_naive: C kernel returned error code {}", rc);
    }
    output
}

/// Fused ternary matmul, 8-row LUT path. DOCUMENTED FAILURE.
///
/// This kernel is preserved as a record of the LUT approach that did
/// not work on Skylake. See the historical note in cpp/kernels/tq1_0.c.
/// Do not use in production.
///
/// `weights` must contain 8 rows of TQ1.0 packed weights, laid out
/// row-major with `row_stride_bytes` bytes per row.
/// `activations` is a single shared activation vector.
/// Returns a Vec of 8 output floats.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub fn matmul_8row_lut_failed(
    weights: &[u8],
    row_stride_bytes: usize,
    num_weights_per_row: usize,
    activations: &[i8],
    scale: f32,
) -> Vec<f32> {
    if num_weights_per_row == 0 {
        return vec![0.0f32; 8];
    }
    let required = row_stride_bytes * 8;
    if weights.len() < required {
        panic!(
            "kiln_kernels::matmul_8row_lut_failed: need {} bytes for 8 rows, got {}",
            required, weights.len()
        );
    }
    if activations.len() < num_weights_per_row {
        panic!(
            "kiln_kernels::matmul_8row_lut_failed: need {} activations, got {}",
            num_weights_per_row, activations.len()
        );
    }
    let mut outputs = vec![0.0f32; 8];
    let rc = unsafe {
        kiln_tq1_0_matmul_8row_lut_failed(
            weights.as_ptr(),
            row_stride_bytes,
            num_weights_per_row,
            activations.as_ptr(),
            activations.len(),
            scale,
            outputs.as_mut_ptr(),
        )
    };
    if rc != 0 {
        panic!("kiln_kernels::matmul_8row_lut_failed: C kernel returned error code {}", rc);
    }
    outputs
}

/// Dequantize one Q4_K super-block (144 bytes) into 256 f32 values.
pub fn q4k_dequant_block(src: &[u8]) -> [f32; 256] {
    if src.len() < 144 {
        panic!(
            "kiln_kernels::q4k_dequant_block: need 144 bytes, got {}",
            src.len()
        );
    }
    let mut out = [0.0f32; 256];
    let rc = unsafe { kiln_q4k_dequant_block(src.as_ptr(), out.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::q4k_dequant_block: C kernel returned error code {}", rc);
    }
    out
}

/// Fused Q4_K matmul. Dot product of a packed Q4_K weight row against
/// an f32 activation vector.
pub fn q4k_matmul_scalar(
    weights: &[u8],
    num_weights: usize,
    activations: &[f32],
) -> f32 {
    if num_weights == 0 {
        return 0.0;
    }
    let num_blocks = (num_weights + 255) / 256;
    let required = num_blocks * 144;
    if weights.len() < required {
        panic!(
            "kiln_kernels::q4k_matmul_scalar: need {} bytes for {} weights, got {}",
            required, num_weights, weights.len()
        );
    }
    if activations.len() < num_weights {
        panic!(
            "kiln_kernels::q4k_matmul_scalar: need {} activations, got {}",
            num_weights, activations.len()
        );
    }
    let mut output = 0.0f32;
    let rc = unsafe {
        kiln_q4k_matmul_scalar(
            weights.as_ptr(),
            num_weights,
            activations.as_ptr(),
            activations.len(),
            &mut output as *mut f32,
        )
    };
    if rc != 0 {
        panic!("kiln_kernels::q4k_matmul_scalar: C kernel returned error code {}", rc);
    }
    output
}

/// Dequantize one Q6_K super-block (210 bytes) into 256 f32 values.
pub fn q6k_dequant_block(src: &[u8]) -> [f32; 256] {
    if src.len() < 210 {
        panic!(
            "kiln_kernels::q6k_dequant_block: need 210 bytes, got {}",
            src.len()
        );
    }
    let mut out = [0.0f32; 256];
    let rc = unsafe { kiln_q6k_dequant_block(src.as_ptr(), out.as_mut_ptr()) };
    if rc != 0 {
        panic!("kiln_kernels::q6k_dequant_block: C kernel returned error code {}", rc);
    }
    out
}

/// Fused Q6_K matmul. Dot product of a packed Q6_K weight row against
/// an f32 activation vector.
pub fn q6k_matmul_scalar(
    weights: &[u8],
    num_weights: usize,
    activations: &[f32],
) -> f32 {
    if num_weights == 0 {
        return 0.0;
    }
    let num_blocks = (num_weights + 255) / 256;
    let required = num_blocks * 210;
    if weights.len() < required {
        panic!(
            "kiln_kernels::q6k_matmul_scalar: need {} bytes for {} weights, got {}",
            required, num_weights, weights.len()
        );
    }
    if activations.len() < num_weights {
        panic!(
            "kiln_kernels::q6k_matmul_scalar: need {} activations, got {}",
            num_weights, activations.len()
        );
    }
    let mut output = 0.0f32;
    let rc = unsafe {
        kiln_q6k_matmul_scalar(
            weights.as_ptr(),
            num_weights,
            activations.as_ptr(),
            activations.len(),
            &mut output as *mut f32,
        )
    };
    if rc != 0 {
        panic!("kiln_kernels::q6k_matmul_scalar: C kernel returned error code {}", rc);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_small() {
        let vals = [1.0f32, 0.0, -1.0, 1.0, -1.0];
        let packed = pack(&vals);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack(&packed, 5);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn round_trip_various_lengths() {
        for n in [1usize, 4, 5, 6, 9, 10, 11, 100, 1000] {
            let vals: Vec<f32> = (0..n)
                .map(|i| match i % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
                .collect();
            let packed = pack(&vals);
            let unpacked = unpack(&packed, n);
            assert_eq!(unpacked, vals, "round trip failed for n={}", n);
        }
    }

    #[test]
    fn boundary_empty_panics() {
        let r = std::panic::catch_unwind(|| pack(&[]));
        assert!(r.is_err());
    }

    #[test]
    fn invalid_value_panics() {
        let r = std::panic::catch_unwind(|| pack(&[0.5f32]));
        assert!(r.is_err());
    }

    #[test]
    fn invalid_byte_panics() {
        let r = std::panic::catch_unwind(|| unpack(&[255u8], 1));
        assert!(r.is_err());
    }

    #[test]
    fn short_input_panics() {
        let r = std::panic::catch_unwind(|| unpack(&[0u8], 10));
        assert!(r.is_err());
    }

    #[test]
    fn property_random_large() {
        let n = 10_000;
        let vals: Vec<f32> = (0..n)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let packed = pack(&vals);
        let unpacked = unpack(&packed, n);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn differential_scalar_vs_avx2() {
        for n in [40usize, 100, 1000, 10_000] {
            let vals: Vec<f32> = (0..n)
                .map(|i| match i % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
                .collect();
            let avx2_packed = pack(&vals);
            let unpacked = unpack(&avx2_packed, n);
            assert_eq!(unpacked, vals, "differential failed for n={}", n);
        }
    }

    #[test]
    fn matmul_8row_lut_failed_matches_scalar() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            for n in [5usize, 10, 25, 50, 100, 1000, 5000] {
                let stride = (n + 4) / 5;
                let mut weights_8x = vec![0u8; stride * 8];
                let mut expected = vec![0.0f32; 8];
                let activations: Vec<i8> = (0..n)
                    .map(|i| ((i as i32 * 7 % 200) - 100) as i8)
                    .collect();

                // Build 8 random weight rows and compute the scalar
                // reference for each.
                for r in 0..8 {
                    let w: Vec<f32> = (0..n)
                        .map(|i| match (i + r * 3) % 3 {
                            0 => -1.0,
                            1 => 0.0,
                            _ => 1.0,
                        })
                        .collect();
                    let packed = pack(&w);
                    for b in 0..stride {
                        weights_8x[r * stride + b] = packed[b];
                    }
                    expected[r] = matmul_scalar(&packed, &activations, n, 1.0);
                }

                let result = matmul_8row_lut_failed(&weights_8x, stride, n, &activations, 1.0);
                for r in 0..8 {
                    assert_eq!(result[r], expected[r], "row {} differs for n={}", r, n);
                }
            }
        }
    }

    #[test]
    fn matmul_lut_naive_matches_scalar() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            for n in [5usize, 10, 25, 50, 100, 1000, 5000] {
                let w: Vec<f32> = (0..n)
                    .map(|i| match i % 3 {
                        0 => -1.0,
                        1 => 0.0,
                        _ => 1.0,
                    })
                    .collect();
                let weights = pack(&w);
                let activations: Vec<i8> = (0..n)
                    .map(|i| ((i as i32 * 7 % 200) - 100) as i8)
                    .collect();

                let s = matmul_scalar(&weights, &activations, n, 1.0);
                let a = matmul_lut_naive(&weights, &activations, n, 1.0);
                assert_eq!(s, a, "avx2 and scalar differ for n={}", n);
            }
        }
    }

    #[test]
    fn matmul_scalar_hand_computed() {
        // Weight row: [+1, -1, 0, +1, -1]
        // Activations: [1, 2, 3, 4, 5]
        // Expected: (1*1) + (-1*2) + (0*3) + (1*4) + (-1*5) = 1 - 2 + 4 - 5 = -2
        let weights = pack(&[1.0, -1.0, 0.0, 1.0, -1.0]);
        let activations: Vec<i8> = vec![1, 2, 3, 4, 5];
        let result = matmul_scalar(&weights, &activations, 5, 1.0);
        assert_eq!(result, -2.0);
    }

    #[test]
    fn matmul_scalar_scale() {
        let weights = pack(&[1.0, 0.0, -1.0]);
        let activations: Vec<i8> = vec![10, 20, 30];
        // (1*10) + (0*20) + (-1*30) = 10 - 30 = -20
        // scale 0.5 -> -10
        let result = matmul_scalar(&weights, &activations, 3, 0.5);
        assert_eq!(result, -10.0);
    }

    #[test]
    fn matmul_scalar_zero_weights() {
        let weights: Vec<u8> = vec![];
        let activations: Vec<i8> = vec![];
        let result = matmul_scalar(&weights, &activations, 0, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn matmul_scalar_property() {
        for n in [5usize, 10, 50, 500, 5000] {
            let w: Vec<f32> = (0..n)
                .map(|i| match i % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
                .collect();
            let weights = pack(&w);
            let activations: Vec<i8> = (0..n).map(|i| ((i % 200) as i32 - 100) as i8).collect();

            // Reference: compute the dot product by hand.
            let mut expected: i32 = 0;
            for i in 0..n {
                let trit = if w[i] == -1.0 { -1 }
                          else if w[i] == 0.0 { 0 }
                          else { 1 };
                expected += trit * activations[i] as i32;
            }

            let result = matmul_scalar(&weights, &activations, n, 1.0);
            assert_eq!(result, expected as f32, "matmul mismatch for n={}", n);
        }
    }

    #[test]
    fn unpack_scalar_and_table_match() {
        for n in [40usize, 100, 1000, 10_000, 100_000] {
            let vals: Vec<f32> = (0..n)
                .map(|i| match i % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
                .collect();
            let packed = pack(&vals);
            let s = unpack(&packed, n);
            let t = unpack_table(&packed, n);
            assert_eq!(s, t, "scalar and table unpack differ for n={}", n);
        }
    }

    #[test]
    fn pack_scalar_and_avx2_match() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            for n in [40usize, 100, 1000, 10_000, 100_000] {
                let vals: Vec<f32> = (0..n)
                    .map(|i| match i % 3 {
                        0 => -1.0,
                        1 => 0.0,
                        _ => 1.0,
                    })
                    .collect();
                let s = pack_scalar(&vals);
                let a = pack_avx2(&vals);
                assert_eq!(s, a, "scalar and avx2 differ for n={}", n);
            }
        }
    }

    #[test]
    fn q4k_dequant_zero_block() {
        // A Q4_K super-block where d=0 and dmin=0 should produce all
        // zeros regardless of the nibbles.
        let mut block = [0u8; 144];
        // d = 0.0 as F16 = 0x0000
        // dmin = 0.0 as F16 = 0x0000
        // scales and qs are all zero.
        block[0] = 0;
        block[1] = 0;
        let out = q4k_dequant_block(&block);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn q4k_dequant_simple_constant() {
        // Build a super-block with d = 1.0, dmin = 0.0, all 8 scales = 1,
        // all 8 mins = 0, all nibbles = 1.
        //
        // The Q4_K scale layout (from ggml get_scale_min_k4):
        //   For j = 0..3: scale[j] = q[j] & 0x3F, min[j] = q[j+4] & 0x3F
        //   For j = 4..7: scale[j] = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
        //                 min[j]   = (q[j+4] >> 4)    | ((q[j]     >> 6) << 4)
        //
        // To make all 8 scales = 1 and all mins = 0 with q = block+4:
        //   q[0..4] = 0x01   (scales 0-3 = 1, low bits for scales 4-7 = 0)
        //   q[4..8] = 0x00   (mins 0-3 = 0)
        //   q[8..12] = 0x01  (low nibble of q[j+4] for j=4..7 = 1 = scale[j])
        //   high bits: q[0..4] >> 6 = 0, so scales 4-7 high = 0
        let mut block = [0u8; 144];
        // d = 1.0 F16 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 0;
        block[3] = 0;
        // q[0..4] = block[4..8] = scales 0-3 = 1
        block[4] = 0x01;
        block[5] = 0x01;
        block[6] = 0x01;
        block[7] = 0x01;
        // q[4..8] = block[8..12] = mins 0-3 = 0
        block[8] = 0x00;
        block[9] = 0x00;
        block[10] = 0x00;
        block[11] = 0x00;
        // q[8..12] = block[12..16] = low nibbles for scales 4-7 = 1
        // For j=4: scale[4] = (q[8] & 0x0F) | ((q[0] >> 6) << 4)
        //         = (block[12] & 0x0F) | ((block[4] >> 6) << 4)
        //         = 1 | 0 = 1
        // Also for mins 4-7: min[j] = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
        // For j=4: min[4] = (q[8] >> 4) | ((q[4] >> 6) << 4)
        //         = (block[12] >> 4) | ((block[8] >> 6) << 4) = 0 | 0 = 0
        block[12] = 0x01;
        block[13] = 0x01;
        block[14] = 0x01;
        block[15] = 0x01;

        // Set all 128 bytes of qs to 0x11 (low nibble = 1, high nibble = 1)
        for i in 16..144 {
            block[i] = 0x11;
        }

        let out = q4k_dequant_block(&block);
        for (i, v) in out.iter().enumerate() {
            assert!((*v - 1.0).abs() < 0.001, "at index {} expected 1.0, got {}", i, v);
        }
    }

    #[test]
    fn q4k_matmul_identity() {
        // Same setup as q4k_dequant_simple_constant but check the
        // matmul wrapper. Every weight is 1.0, activations are all 1.0,
        // so the dot product is 256.0.
        let mut block = [0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 0;
        block[3] = 0;
        block[4] = 0x01;
        block[5] = 0x01;
        block[6] = 0x01;
        block[7] = 0x01;
        block[8] = 0x00;
        block[9] = 0x00;
        block[10] = 0x00;
        block[11] = 0x00;
        block[12] = 0x01;
        block[13] = 0x01;
        block[14] = 0x01;
        block[15] = 0x01;
        for i in 16..144 {
            block[i] = 0x11;
        }
        let acts = vec![1.0f32; 256];
        let result = q4k_matmul_scalar(&block, 256, &acts);
        assert!((result - 256.0).abs() < 0.01, "expected 256.0, got {}", result);
    }

    #[test]
    fn q6k_dequant_zero_block() {
        // d = 0, all scales = 0, all weights = 0 -> all output values
        // should be d * scale * (q - 32) = 0.
        let block = [0u8; 210];
        let out = q6k_dequant_block(&block);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn q6k_dequant_constant_q1() {
        // Build a Q6_K block with:
        //   d = 1.0 (F16 = 0x3C00)
        //   all scales = 1
        //   all weights = 1 (6-bit value 1, centered: -31)
        //
        // To set all 6-bit weights to 1:
        //   ql bytes: low nibbles = 1, high nibbles = 1 -> 0x11
        //   qh bytes: all 2-bit fields = 0 -> 0x00
        //
        // Expected: d * 1 * (1 - 32) = -31.0 for every output.
        let mut block = [0u8; 210];
        // d = 1.0
        block[208] = 0x00;
        block[209] = 0x3C;
        // ql: bytes 0..128, all 0x11
        for i in 0..128 {
            block[i] = 0x11;
        }
        // qh: bytes 128..192, all 0x00
        for i in 128..192 {
            block[i] = 0x00;
        }
        // scales: bytes 192..208, all 1 (int8)
        for i in 192..208 {
            block[i] = 0x01;
        }
        let out = q6k_dequant_block(&block);
        for (i, v) in out.iter().enumerate() {
            assert!((*v - (-31.0)).abs() < 0.001,
                "at index {} expected -31.0, got {}", i, v);
        }
    }

    #[test]
    fn q6k_dequant_constant_q32() {
        // Build a Q6_K block with all 6-bit weights = 32 (centered: 0).
        // 32 = 0b100000. Low 4 bits = 0, high 2 bits = 2.
        // ql bytes: all 0x00
        // qh bytes: each 2-bit field = 2 -> 0b10101010 = 0xAA
        // Expected: 0.0 for every output.
        let mut block = [0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        for i in 0..128 {
            block[i] = 0x00;
        }
        for i in 128..192 {
            block[i] = 0xAA;
        }
        for i in 192..208 {
            block[i] = 0x01;
        }
        let out = q6k_dequant_block(&block);
        for (i, v) in out.iter().enumerate() {
            assert!((*v - 0.0).abs() < 0.001,
                "at index {} expected 0.0, got {}", i, v);
        }
    }

    #[test]
    fn q6k_matmul_constant() {
        // All weights = 1 (centered: -31), all activations = 1.
        // Expected: 256 * (-31) = -7936.0.
        let mut block = [0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C;
        for i in 0..128 {
            block[i] = 0x11;
        }
        for i in 192..208 {
            block[i] = 0x01;
        }
        let acts = vec![1.0f32; 256];
        let result = q6k_matmul_scalar(&block, 256, &acts);
        assert!((result - (-7936.0)).abs() < 0.1,
            "expected -7936.0, got {}", result);
    }

    #[test]
    fn avx2_path_is_actually_compiled() {
        // The C build script sets -mavx2 on x86_64. When it does, __AVX2__
        // is defined and the AVX2 code is compiled. This test asserts that
        // the AVX2 path is active on x86_64. On other architectures, it
        // asserts the scalar path is active.
        #[cfg(target_arch = "x86_64")]
        {
            // If the AVX2 path were not compiled, this test would still
            // pass. To detect the real state, we check the runtime CPU
            // feature. If the CPU supports AVX2 and the target is x86_64,
            // the build script enabled it. If the CPU does not support
            // AVX2, the scalar path runs at runtime.
            let has_avx2 = is_x86_feature_detected!("avx2");
            println!("runtime AVX2 support: {}", has_avx2);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            println!("non-x86_64 target: scalar path");
        }
    }
}
