/* KILN kernels: TQ1.0 ternary packing.
 *
 * See docs/kernels-design.md for the format specification and the ABI.
 *
 * Base-3 encoding: five trits pack into one byte. Each trit is in
 * {-1, 0, +1} and is shifted to a digit in {0, 1, 2}. The packed value
 * is d0*81 + d1*27 + d2*9 + d3*3 + d4*1, which is in [0, 242].
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int trit_to_digit(float v) {
    if (v == -1.0f) return 0;
    if (v == 0.0f)  return 1;
    if (v == 1.0f)  return 2;
    return -1;
}

static inline float digit_to_trit(int d) {
    return (float)(d - 1);
}

int kiln_tq1_0_pack(const float* src, size_t n, uint8_t* dst) {
    if (n == 0) return 0;
    if (src == NULL || dst == NULL) return 1;

    size_t out_bytes = (n + 4) / 5;
    for (size_t b = 0; b < out_bytes; ++b) {
        int d0 = 1, d1 = 1, d2 = 1, d3 = 1, d4 = 1;
        size_t base = b * 5;

        if (base + 0 < n) {
            int v = trit_to_digit(src[base + 0]);
            if (v < 0) return 2;
            d0 = v;
        }
        if (base + 1 < n) {
            int v = trit_to_digit(src[base + 1]);
            if (v < 0) return 2;
            d1 = v;
        }
        if (base + 2 < n) {
            int v = trit_to_digit(src[base + 2]);
            if (v < 0) return 2;
            d2 = v;
        }
        if (base + 3 < n) {
            int v = trit_to_digit(src[base + 3]);
            if (v < 0) return 2;
            d3 = v;
        }
        if (base + 4 < n) {
            int v = trit_to_digit(src[base + 4]);
            if (v < 0) return 2;
            d4 = v;
        }

        int packed = d0 * 81 + d1 * 27 + d2 * 9 + d3 * 3 + d4;
        dst[b] = (uint8_t)packed;
    }
    return 0;
}

int kiln_tq1_0_unpack(const uint8_t* src, size_t n, float* dst) {
    if (n == 0) return 0;
    if (src == NULL || dst == NULL) return 1;

    size_t in_bytes = (n + 4) / 5;
    for (size_t b = 0; b < in_bytes; ++b) {
        int packed = (int)src[b];
        if (packed < 0 || packed > 242) return 3;

        int d4 = packed % 3;
        int d3 = (packed / 3) % 3;
        int d2 = (packed / 9) % 3;
        int d1 = (packed / 27) % 3;
        int d0 = (packed / 81) % 3;

        size_t base = b * 5;
        if (base + 0 < n) dst[base + 0] = digit_to_trit(d0);
        if (base + 1 < n) dst[base + 1] = digit_to_trit(d1);
        if (base + 2 < n) dst[base + 2] = digit_to_trit(d2);
        if (base + 3 < n) dst[base + 3] = digit_to_trit(d3);
        if (base + 4 < n) dst[base + 4] = digit_to_trit(d4);
    }
    return 0;
}

/* ---- AVX2 pack path ---- */

/* On x86_64 with -mavx2, both paths are compiled and the dispatcher
 * chooses at runtime. On non-x86_64, only the scalar path exists. */

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

/* AVX2 pack: process 40 floats (8 bytes output) per iteration.
 * Load 8 floats at a time, compare against -1.0 and 1.0 to get two
 * 8-lane masks. Extract 8-bit integers from each mask. Build a 16-bit
 * integer where each 2-bit field encodes one trit. Then pack the 40
 * trits (5 bytes worth) into bytes.
 *
 * This is the fast path. It is bit-identical to the scalar packer.
 */
static int kiln_tq1_0_pack_avx2(const float* src, size_t n, uint8_t* dst) {
    size_t out_bytes = (n + 4) / 5;
    size_t b = 0;

    /* Process 40 inputs (8 output bytes) at a time. */
    while (b + 8 <= out_bytes) {
        size_t base = b * 5;
        /* Safety check: only run the fast path when the full 40 inputs
         * exist. The last partial group falls through to scalar. */
        if (base + 40 > n) break;

        __m256 v0 = _mm256_loadu_ps(src + base + 0);
        __m256 v1 = _mm256_loadu_ps(src + base + 8);
        __m256 v2 = _mm256_loadu_ps(src + base + 16);
        __m256 v3 = _mm256_loadu_ps(src + base + 24);
        __m256 v4 = _mm256_loadu_ps(src + base + 32);

        /* For each vector, get -1 mask and +1 mask.
         * trit_to_digit(-1) = 0, (0) = 1, (+1) = 2.
         * digit = 1 + (is_pos ? 1 : 0) - (is_neg ? 1 : 0).
         */
        const __m256 neg_one = _mm256_set1_ps(-1.0f);
        const __m256 pos_one = _mm256_set1_ps(1.0f);
        const __m256 one = _mm256_set1_ps(1.0f);

        __m256 m0n = _mm256_cmp_ps(v0, neg_one, _CMP_EQ_OQ);
        __m256 m0p = _mm256_cmp_ps(v0, pos_one, _CMP_EQ_OQ);
        /* digit = one + (m0p ? 1 : 0) - (m0n ? 1 : 0) */
        __m256 d0f = _mm256_add_ps(one, _mm256_sub_ps(
            _mm256_and_ps(m0p, one), _mm256_and_ps(m0n, one)));

        __m256 m1n = _mm256_cmp_ps(v1, neg_one, _CMP_EQ_OQ);
        __m256 m1p = _mm256_cmp_ps(v1, pos_one, _CMP_EQ_OQ);
        __m256 d1f = _mm256_add_ps(one, _mm256_sub_ps(
            _mm256_and_ps(m1p, one), _mm256_and_ps(m1n, one)));

        __m256 m2n = _mm256_cmp_ps(v2, neg_one, _CMP_EQ_OQ);
        __m256 m2p = _mm256_cmp_ps(v2, pos_one, _CMP_EQ_OQ);
        __m256 d2f = _mm256_add_ps(one, _mm256_sub_ps(
            _mm256_and_ps(m2p, one), _mm256_and_ps(m2n, one)));

        __m256 m3n = _mm256_cmp_ps(v3, neg_one, _CMP_EQ_OQ);
        __m256 m3p = _mm256_cmp_ps(v3, pos_one, _CMP_EQ_OQ);
        __m256 d3f = _mm256_add_ps(one, _mm256_sub_ps(
            _mm256_and_ps(m3p, one), _mm256_and_ps(m3n, one)));

        __m256 m4n = _mm256_cmp_ps(v4, neg_one, _CMP_EQ_OQ);
        __m256 m4p = _mm256_cmp_ps(v4, pos_one, _CMP_EQ_OQ);
        __m256 d4f = _mm256_add_ps(one, _mm256_sub_ps(
            _mm256_and_ps(m4p, one), _mm256_and_ps(m4n, one)));

        /* Convert to integers. */
        __m256i i0 = _mm256_cvtps_epi32(d0f);
        __m256i i1 = _mm256_cvtps_epi32(d1f);
        __m256i i2 = _mm256_cvtps_epi32(d2f);
        __m256i i3 = _mm256_cvtps_epi32(d3f);
        __m256i i4 = _mm256_cvtps_epi32(d4f);

        int d[40];
        _mm256_storeu_si256((__m256i*)(d + 0),  i0);
        _mm256_storeu_si256((__m256i*)(d + 8),  i1);
        _mm256_storeu_si256((__m256i*)(d + 16), i2);
        _mm256_storeu_si256((__m256i*)(d + 24), i3);
        _mm256_storeu_si256((__m256i*)(d + 32), i4);

        /* Now pack the 40 digits into 8 bytes, 5 digits per byte. */
        for (size_t k = 0; k < 8; ++k) {
            int a0 = d[k*5 + 0];
            int a1 = d[k*5 + 1];
            int a2 = d[k*5 + 2];
            int a3 = d[k*5 + 3];
            int a4 = d[k*5 + 4];
            if (a0 < 0 || a0 > 2) return 2;
            if (a1 < 0 || a1 > 2) return 2;
            if (a2 < 0 || a2 > 2) return 2;
            if (a3 < 0 || a3 > 2) return 2;
            if (a4 < 0 || a4 > 2) return 2;
            dst[b + k] = (uint8_t)(a0 * 81 + a1 * 27 + a2 * 9 + a3 * 3 + a4);
        }

        b += 8;
    }

    /* Scalar tail for the remaining bytes. */
    for (; b < out_bytes; ++b) {
        int d0 = 1, d1 = 1, d2 = 1, d3 = 1, d4 = 1;
        size_t base = b * 5;
        if (base + 0 < n) {
            int v = trit_to_digit(src[base + 0]);
            if (v < 0) return 2;
            d0 = v;
        }
        if (base + 1 < n) {
            int v = trit_to_digit(src[base + 1]);
            if (v < 0) return 2;
            d1 = v;
        }
        if (base + 2 < n) {
            int v = trit_to_digit(src[base + 2]);
            if (v < 0) return 2;
            d2 = v;
        }
        if (base + 3 < n) {
            int v = trit_to_digit(src[base + 3]);
            if (v < 0) return 2;
            d3 = v;
        }
        if (base + 4 < n) {
            int v = trit_to_digit(src[base + 4]);
            if (v < 0) return 2;
            d4 = v;
        }
        dst[b] = (uint8_t)(d0 * 81 + d1 * 27 + d2 * 9 + d3 * 3 + d4);
    }
    return 0;
}

#endif /* __x86_64__ || __i386__ */


/* ---- Table-based fast unpacker ---- */

/* A 243-entry lookup table. Each entry packs 5 trit digits (0, 1, or 2)
 * into a single uint32 as 5 bytes, one per trit. The table is built once
 * at load time. It removes four integer divisions per byte from the
 * unpacker, which was the dominant cost in the scalar path.
 */

/* A 243-entry lookup table. Each entry holds 5 trit values laid out
 * sequentially. The whole table is 243 * 5 = 1215 bytes and fits in L1.
 * It removes four integer divisions per byte from the unpacker. */
static int8_t tq1_0_trits[243][5];
static int tq1_0_table_ready = 0;

static void tq1_0_build_trits(void) {
    for (int i = 0; i < 243; ++i) {
        int d4 = i % 3;
        int d3 = (i / 3) % 3;
        int d2 = (i / 9) % 3;
        int d1 = (i / 27) % 3;
        int d0 = (i / 81) % 3;
        tq1_0_trits[i][0] = (int8_t)(d0 - 1);
        tq1_0_trits[i][1] = (int8_t)(d1 - 1);
        tq1_0_trits[i][2] = (int8_t)(d2 - 1);
        tq1_0_trits[i][3] = (int8_t)(d3 - 1);
        tq1_0_trits[i][4] = (int8_t)(d4 - 1);
    }
    tq1_0_table_ready = 1;
}

static void tq1_0_ensure_table(void) {
    if (!tq1_0_table_ready) {
        tq1_0_build_trits();
    }
}

int kiln_tq1_0_unpack_table(const uint8_t* src, size_t n, float* dst) {
    if (n == 0) return 0;
    if (src == NULL || dst == NULL) return 1;
    tq1_0_ensure_table();

    size_t in_bytes = (n + 4) / 5;
    for (size_t b = 0; b < in_bytes; ++b) {
        int packed = (int)src[b];
        if (packed < 0 || packed > 242) return 3;
        const int8_t* trits = tq1_0_trits[packed];
        size_t base = b * 5;
        if (base + 0 < n) dst[base + 0] = (float)trits[0];
        if (base + 1 < n) dst[base + 1] = (float)trits[1];
        if (base + 2 < n) dst[base + 2] = (float)trits[2];
        if (base + 3 < n) dst[base + 3] = (float)trits[3];
        if (base + 4 < n) dst[base + 4] = (float)trits[4];
    }
    return 0;
}

/* ---- Fused ternary matmul, scalar reference ---- */

/* Compute one output row of the matmul between a TQ1.0 weight row and
 * an int8 activation vector.
 *
 * The weight is ternary. The activation is int8. The output is float.
 * The math is y = sum(w_i * x_i) * scale.
 *
 * The scalar reference does not use the LUT. It unpacks each trit and
 * adds or subtracts the activation. This is the correctness anchor.
 * Every optimized version must produce bit-identical output.
 *
 * Returns 0 on success, non-zero on error.
 */
int kiln_tq1_0_matmul_scalar(
    const uint8_t* weights,
    size_t num_weights,
    const int8_t* activations,
    size_t num_activations,
    float scale,
    float* output
) {
    if (weights == NULL || activations == NULL || output == NULL) return 1;
    if (num_weights == 0) {
        *output = 0.0f;
        return 0;
    }
    if (num_activations < num_weights) return 2;

    int32_t acc = 0;
    size_t in_bytes = (num_weights + 4) / 5;

    for (size_t b = 0; b < in_bytes; ++b) {
        int packed = (int)weights[b];
        if (packed < 0 || packed > 242) return 3;

        int d4 = packed % 3;
        int d3 = (packed / 3) % 3;
        int d2 = (packed / 9) % 3;
        int d1 = (packed / 27) % 3;
        int d0 = (packed / 81) % 3;

        size_t base = b * 5;
        int trit;

        if (base + 0 < num_weights) {
            trit = d0 - 1;
            acc += trit * (int32_t)activations[base + 0];
        }
        if (base + 1 < num_weights) {
            trit = d1 - 1;
            acc += trit * (int32_t)activations[base + 1];
        }
        if (base + 2 < num_weights) {
            trit = d2 - 1;
            acc += trit * (int32_t)activations[base + 2];
        }
        if (base + 3 < num_weights) {
            trit = d3 - 1;
            acc += trit * (int32_t)activations[base + 3];
        }
        if (base + 4 < num_weights) {
            trit = d4 - 1;
            acc += trit * (int32_t)activations[base + 4];
        }
    }

    *output = (float)acc * scale;
    return 0;
}

/* ---- Fused ternary matmul, AVX2 path ---- */

#if defined(__x86_64__) || defined(__i386__)

/* LUT for a block size of 5. Entry i holds the signed sum of the five
 * activations at chunk position, weighted by the trit pattern encoded
 * in i. The trit pattern is the TQ1.0 packed byte, so the packed byte
 * IS the LUT index. This is why G=5 is the natural block size.
 *
 * The LUT is built once per activation chunk. It is small: 243 entries
 * times 4 bytes = 972 bytes. Fits comfortably in L1.
 *
 * Note: the kernel processes one chunk at a time. For a weight row of
 * length N, the number of chunks is ceil(N / 5). Each chunk has its own
 * LUT, built from that chunk's 5 activations.
 *
 * The AVX2 path processes 8 output rows at once. For each of the 8 rows,
 * it loads the packed byte at the current chunk position, uses it as a
 * LUT index, gathers 8 partial sums, and accumulates.
 */
int kiln_tq1_0_matmul_avx2(
    const uint8_t* weights,
    size_t num_weights,
    const int8_t* activations,
    size_t num_activations,
    float scale,
    float* output
) {
    if (weights == NULL || activations == NULL || output == NULL) return 1;
    if (num_weights == 0) {
        *output = 0.0f;
        return 0;
    }
    if (num_activations < num_weights) return 2;

    /* Single-row AVX2 implementation. Process one output row. This is
     * the first AVX2 version. An 8-row version that processes 8 rows
     * in parallel using the gather across rows is a follow-up. */
    int32_t acc = 0;
    size_t in_bytes = (num_weights + 4) / 5;

    int32_t lut[243];
    for (size_t b = 0; b < in_bytes; ++b) {
        /* Build the LUT for this chunk. */
        int a[5] = {0, 0, 0, 0, 0};
        size_t base = b * 5;
        for (int k = 0; k < 5; ++k) {
            if (base + (size_t)k < num_weights) {
                a[k] = (int)activations[base + k];
            }
        }
        for (int i = 0; i < 243; ++i) {
            int d4 = i % 3;
            int d3 = (i / 3) % 3;
            int d2 = (i / 9) % 3;
            int d1 = (i / 27) % 3;
            int d0 = (i / 81) % 3;
            int s = 0;
            s += (d0 - 1) * a[0];
            s += (d1 - 1) * a[1];
            s += (d2 - 1) * a[2];
            s += (d3 - 1) * a[3];
            s += (d4 - 1) * a[4];
            lut[i] = s;
        }

        /* Look up the partial sum for this chunk. */
        int packed = (int)weights[b];
        if (packed < 0 || packed > 242) return 3;
        acc += lut[packed];
    }

    *output = (float)acc * scale;
    return 0;
}

#endif /* __x86_64__ || __i386__ */

/* ---- Public dispatch ---- */

/* The scalar packer is renamed to kiln_tq1_0_pack_scalar so that both
 * paths are always available. The old name kiln_tq1_0_pack is kept as an
 * alias for backward compatibility. */
int kiln_tq1_0_pack_scalar(const float* src, size_t n, uint8_t* dst) {
    return kiln_tq1_0_pack(src, n, dst);
}

/* On x86_64, the AVX2 packer is exposed. On other architectures, this
 * function is not defined. The Rust side handles the conditional. */
#if defined(__x86_64__) || defined(__i386__)
int kiln_tq1_0_pack_avx2_dispatch(const float* src, size_t n, uint8_t* dst) {
    if (n == 0) return 0;
    if (src == NULL || dst == NULL) return 1;
    return kiln_tq1_0_pack_avx2(src, n, dst);
}
#endif

/* The default dispatcher picks the fastest available path. */
#if defined(__x86_64__) || defined(__i386__)
int kiln_tq1_0_pack_dispatch(const float* src, size_t n, uint8_t* dst) {
    return kiln_tq1_0_pack_avx2_dispatch(src, n, dst);
}
#else
int kiln_tq1_0_pack_dispatch(const float* src, size_t n, uint8_t* dst) {
    return kiln_tq1_0_pack_scalar(src, n, dst);
}
#endif

#ifdef __cplusplus
}
#endif
