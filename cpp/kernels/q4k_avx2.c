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

#endif /* __AVX2__ */
