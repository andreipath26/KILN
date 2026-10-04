/* KILN: AVX2 Q4_K block dequantizer.
 *
 * Produces the same 256 floats as kiln_q4k_dequant_block, faster.
 * The inner loop reads 32 bytes (64 nibbles) at a time and produces
 * 64 f32 outputs, matching the ggml byte layout exactly.
 *
 * Only compiled when __AVX2__ is defined.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>

/* Forward declaration. Defined in q4k.c. */
int kiln_q4k_matmul_scalar(
    const uint8_t* weights, size_t num_weights,
    const float* activations, size_t num_activations,
    float* output);

static float q4k_avx2_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) { uint32_t b = sign<<31; float f; memcpy(&f,&b,4); return f; }
        while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
        exp += 1; mant &= 0x3FFu;
    } else if (exp == 0x1Fu) {
        uint32_t b = (sign<<31)|0x7F800000u|(mant<<13); float f; memcpy(&f,&b,4); return f;
    }
    uint32_t fe = exp + (127 - 15);
    uint32_t b = (sign<<31)|(fe<<23)|(mant<<13); float f; memcpy(&f,&b,4); return f;
}

static void q4k_avx2_unpack_scales(const uint8_t* q, uint8_t* sc, uint8_t* mn) {
    for (int j = 0; j < 4; ++j) {
        sc[j] = q[j] & 0x3F;
        mn[j] = q[j + 4] & 0x3F;
    }
    for (int j = 4; j < 8; ++j) {
        sc[j] = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        mn[j] = (uint8_t)((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4));
    }
}

int kiln_q4k_dequant_block_avx2(const uint8_t* src, float* dst) {
    if (!src || !dst) return 1;

    float d    = q4k_avx2_f16_to_f32((uint16_t)src[0] | ((uint16_t)src[1] << 8));
    float dmin = q4k_avx2_f16_to_f32((uint16_t)src[2] | ((uint16_t)src[3] << 8));
    uint8_t sc[8], mn[8];
    q4k_avx2_unpack_scales(src + 4, sc, mn);

    const uint8_t* qs = src + 16;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    for (int g = 0; g < 4; ++g) {
        __m256i bytes = _mm256_loadu_si256((const __m256i*)(qs + g * 32));

        /* Low nibbles of all 32 bytes -> 32 values 0..15. */
        __m256i lo_bytes = _mm256_and_si256(bytes, mask_0f);
        /* High nibbles -> 32 values 0..15. */
        __m256i hi_bytes = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask_0f);

        /* Widen to 32-bit ints. _mm256_cvtepu8_epi32 takes the low 8 bytes
         * of a 128-bit lane and produces 8 dwords. We need 4 calls per
         * 32 bytes: two for the low half of lo_bytes, two for the
         * low half of hi_bytes. */
        __m128i lo_lo128 = _mm256_castsi256_si128(lo_bytes);
        __m128i lo_hi128 = _mm256_extracti128_si256(lo_bytes, 1);
        __m128i hi_lo128 = _mm256_castsi256_si128(hi_bytes);
        __m128i hi_hi128 = _mm256_extracti128_si256(hi_bytes, 1);

        __m256i lo0 = _mm256_cvtepu8_epi32(lo_lo128);          /* bytes 0..7   */
        __m256i lo1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_lo128, 8)); /* 8..15 */
        __m256i lo2 = _mm256_cvtepu8_epi32(lo_hi128);          /* 16..23 */
        __m256i lo3 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_hi128, 8)); /* 24..31 */
        __m256i hi0 = _mm256_cvtepu8_epi32(hi_lo128);
        __m256i hi1 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_lo128, 8));
        __m256i hi2 = _mm256_cvtepu8_epi32(hi_hi128);
        __m256i hi3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_hi128, 8));

        __m256 s_lo = _mm256_set1_ps(d * (float)sc[2 * g]);
        __m256 m_lo = _mm256_set1_ps(dmin * (float)mn[2 * g]);
        __m256 s_hi = _mm256_set1_ps(d * (float)sc[2 * g + 1]);
        __m256 m_hi = _mm256_set1_ps(dmin * (float)mn[2 * g + 1]);

        /* Output layout for group g:
         *   dst[g*64 +  0 .. g*64 + 31] <- low nibbles, scale sc[2g]
         *   dst[g*64 + 32 .. g*64 + 63] <- high nibbles, scale sc[2g+1]
         *   low nibble l -> position l, high nibble l -> position 32+l
         *   (l is the byte index within the group, 0..31)
         *
         * Each cvtepu8_epi32 call gave us 8 consecutive byte values, so
         * lo0 holds nibble values for bytes 0..7, lo1 for 8..15, etc. */

        __m256 w_lo0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo0), s_lo), m_lo);
        __m256 w_lo1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo1), s_lo), m_lo);
        __m256 w_lo2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo2), s_lo), m_lo);
        __m256 w_lo3 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo3), s_lo), m_lo);
        __m256 w_hi0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi0), s_hi), m_hi);
        __m256 w_hi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi1), s_hi), m_hi);
        __m256 w_hi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi2), s_hi), m_hi);
        __m256 w_hi3 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi3), s_hi), m_hi);

        float* out = dst + g * 64;
        _mm256_storeu_ps(out +  0, w_lo0);
        _mm256_storeu_ps(out +  8, w_lo1);
        _mm256_storeu_ps(out + 16, w_lo2);
        _mm256_storeu_ps(out + 24, w_lo3);
        _mm256_storeu_ps(out + 32, w_hi0);
        _mm256_storeu_ps(out + 40, w_hi1);
        _mm256_storeu_ps(out + 48, w_hi2);
        _mm256_storeu_ps(out + 56, w_hi3);
    }
    return 0;
}

/* Fused AVX2 Q4_K matmul.
 *
 * Dequant and dot in the same loop. No stack array, no per-block FFI.
 * The scalar path already does this fusion; this is the vectorized
 * version. The accumulator stays in a __m256 register across the whole
 * row and is horizontal-summed once at the end.
 *
 * Layout note: the activations x are indexed in natural order. Within
 * group g, the low nibbles of the 32 bytes go to output positions
 * [g*64 .. g*64+31] and the high nibbles go to [g*64+32 .. g*64+63].
 */
int kiln_q4k_matmul_avx2(
    const uint8_t* weights, size_t num_weights,
    const float* activations, size_t num_activations,
    float* output)
{
    if (!weights || !activations || !output) return 1;
    if (num_weights == 0) { *output = 0.0f; return 0; }
    if (num_activations < num_weights) return 2;

    size_t num_blocks = (num_weights + 255) / 256;
    __m256 accv = _mm256_setzero_ps();
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* src = weights + b * 144;
        float d    = q4k_avx2_f16_to_f32((uint16_t)src[0] | ((uint16_t)src[1] << 8));
        float dmin = q4k_avx2_f16_to_f32((uint16_t)src[2] | ((uint16_t)src[3] << 8));
        uint8_t sc[8], mn[8];
        q4k_avx2_unpack_scales(src + 4, sc, mn);
        const uint8_t* qs = src + 16;

        size_t base = b * 256;
        int full = (base + 256 <= num_weights);
        size_t count = full ? 256 : (num_weights - base);
        /* For the common case, count is always 256 because hidden,
         * inter, q_dim, kv_dim are multiples of 256. If not full, we
         * fall through to a scalar tail per group. */

        for (int g = 0; g < 4; ++g) {
            __m256i bytes = _mm256_loadu_si256((const __m256i*)(qs + g * 32));

            __m256i lo_bytes = _mm256_and_si256(bytes, mask_0f);
            __m256i hi_bytes = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask_0f);

            __m128i lo_lo128 = _mm256_castsi256_si128(lo_bytes);
            __m128i lo_hi128 = _mm256_extracti128_si256(lo_bytes, 1);
            __m128i hi_lo128 = _mm256_castsi256_si128(hi_bytes);
            __m128i hi_hi128 = _mm256_extracti128_si256(hi_bytes, 1);

            __m256i lo0 = _mm256_cvtepu8_epi32(lo_lo128);
            __m256i lo1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_lo128, 8));
            __m256i lo2 = _mm256_cvtepu8_epi32(lo_hi128);
            __m256i lo3 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_hi128, 8));
            __m256i hi0 = _mm256_cvtepu8_epi32(hi_lo128);
            __m256i hi1 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_lo128, 8));
            __m256i hi2 = _mm256_cvtepu8_epi32(hi_hi128);
            __m256i hi3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_hi128, 8));

            __m256 s_lo = _mm256_set1_ps(d * (float)sc[2 * g]);
            __m256 m_lo = _mm256_set1_ps(dmin * (float)mn[2 * g]);
            __m256 s_hi = _mm256_set1_ps(d * (float)sc[2 * g + 1]);
            __m256 m_hi = _mm256_set1_ps(dmin * (float)mn[2 * g + 1]);

            /* Dequantized weights, in the group-local order:
             *   lo0..lo3 = positions  0..31 (low nibbles)
             *   hi0..hi3 = positions 32..63 (high nibbles) */
            __m256 w_lo0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo0), s_lo), m_lo);
            __m256 w_lo1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo1), s_lo), m_lo);
            __m256 w_lo2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo2), s_lo), m_lo);
            __m256 w_lo3 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo3), s_lo), m_lo);
            __m256 w_hi0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi0), s_hi), m_hi);
            __m256 w_hi1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi1), s_hi), m_hi);
            __m256 w_hi2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi2), s_hi), m_hi);
            __m256 w_hi3 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi3), s_hi), m_hi);

            /* Load the corresponding activations. If the block is not
             * full, load whatever fits and zero the rest. This is the
             * slow edge case that only fires on partial final blocks. */
            const float* x = activations + base + g * 64;
            int group_room = (int)count - g * 64;
            if (group_room >= 64) {
                __m256 a0 = _mm256_loadu_ps(x +  0);
                __m256 a1 = _mm256_loadu_ps(x +  8);
                __m256 a2 = _mm256_loadu_ps(x + 16);
                __m256 a3 = _mm256_loadu_ps(x + 24);
                __m256 a4 = _mm256_loadu_ps(x + 32);
                __m256 a5 = _mm256_loadu_ps(x + 40);
                __m256 a6 = _mm256_loadu_ps(x + 48);
                __m256 a7 = _mm256_loadu_ps(x + 56);
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_lo0, a0));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_lo1, a1));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_lo2, a2));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_lo3, a3));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_hi0, a4));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_hi1, a5));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_hi2, a6));
                accv = _mm256_add_ps(accv, _mm256_mul_ps(w_hi3, a7));
            } else {
                /* Partial group. Scalar fallback for correctness. */
                float wbuf[64];
                _mm256_storeu_ps(wbuf +  0, w_lo0);
                _mm256_storeu_ps(wbuf +  8, w_lo1);
                _mm256_storeu_ps(wbuf + 16, w_lo2);
                _mm256_storeu_ps(wbuf + 24, w_lo3);
                _mm256_storeu_ps(wbuf + 32, w_hi0);
                _mm256_storeu_ps(wbuf + 40, w_hi1);
                _mm256_storeu_ps(wbuf + 48, w_hi2);
                _mm256_storeu_ps(wbuf + 56, w_hi3);
                float acc_s = 0.0f;
                for (int i = 0; i < group_room; ++i) {
                    acc_s += wbuf[i] * x[i];
                }
                accv = _mm256_add_ps(accv, _mm256_set1_ps(acc_s));
            }
        }
    }

    /* Horizontal sum of the 8-float accumulator. */
    __m128 lo128 = _mm256_castps256_ps128(accv);
    __m128 hi128 = _mm256_extractf128_ps(accv, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    *output = _mm_cvtss_f32(sums);
    return 0;
}

/* Dispatch. */
int kiln_q4k_matmul_dispatch(
    const uint8_t* weights, size_t num_weights,
    const float* activations, size_t num_activations,
    float* output)
{
    if (__builtin_cpu_supports("avx2")) {
        return kiln_q4k_matmul_avx2(weights, num_weights, activations, num_activations, output);
    }
    return kiln_q4k_matmul_scalar(weights, num_weights, activations, num_activations, output);
}

#endif /* __AVX2__ */
