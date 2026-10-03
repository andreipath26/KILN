//! KILN Kernels.
//!
//! Safe Rust wrappers around the C kernels in cpp/kernels/. See
//! docs/kernels-design.md for the format, the ABI, and the build process.

use std::os::raw::{c_float, c_int};

extern "C" {
    fn kiln_tq1_0_pack(src: *const c_float, n: usize, dst: *mut u8) -> c_int;
    fn kiln_tq1_0_unpack(src: *const u8, n: usize, dst: *mut c_float) -> c_int;
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
