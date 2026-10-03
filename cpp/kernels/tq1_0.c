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

#if defined(__AVX2__) || defined(KILN_FORCE_AVX2)

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

#endif /* __AVX2__ || KILN_FORCE_AVX2 */


/* ---- Public dispatch ---- */

#if defined(__AVX2__) || defined(KILN_FORCE_AVX2)
int kiln_tq1_0_pack_dispatch(const float* src, size_t n, uint8_t* dst) {
    if (n == 0) return 0;
    if (src == NULL || dst == NULL) return 1;
    return kiln_tq1_0_pack_avx2(src, n, dst);
}
#else
int kiln_tq1_0_pack_dispatch(const float* src, size_t n, uint8_t* dst) {
    return kiln_tq1_0_pack(src, n, dst);
}
#endif

#ifdef __cplusplus
}
#endif
