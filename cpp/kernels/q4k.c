/* KILN kernel: Q4_K dequantization and fused matmul.
 *
 * See docs/q4k-design.md for the format specification.
 */

#include "q4k.h"
#include <string.h>

/* Convert an IEEE 754 half-precision float (F16) to single-precision
 * float (F32). */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero. */
            uint32_t bits = sign << 31;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Subnormal. Normalize. */
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= 0x3FFu;
    } else if (exp == 0x1Fu) {
        /* Infinity or NaN. */
        uint32_t bits = (sign << 31) | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    uint32_t f32_exp = exp + (127 - 15);
    uint32_t bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Read a little-endian uint16 from bytes. */
static uint16_t read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

/* Unpack the 6-bit scales and mins from the 12-byte field.
 *
 * The layout is: the low 4 bits of scales 0-3 are the low nibbles of
 * bytes 0-3. The low 4 bits of scales 4-7 are the low nibbles of bytes
 * 4-7. The high nibbles of bytes 0-7 hold the low 4 bits of the mins.
 * Bytes 8-11 hold the high 2 bits of all 16 values, 4 values per byte.
 *
 * This layout matches ggml's get_scale_min_k4 in ggml-quants.c.
 */
static void unpack_scales(
    const uint8_t* q,
    uint8_t* out_scales,
    uint8_t* out_mins)
{
    /* The canonical layout from ggml's get_scale_min_k4.
     *
     * For j = 0..3:
     *   scale[j] = q[j] & 0x3F
     *   min[j]   = q[j + 4] & 0x3F
     *
     * For j = 4..7:
     *   scale[j] = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
     *   min[j]   = (q[j + 4] >> 4)    | ((q[j]     >> 6) << 4)
     */
    for (int j = 0; j < 4; ++j) {
        out_scales[j] = q[j] & 0x3F;
        out_mins[j]   = q[j + 4] & 0x3F;
    }
    for (int j = 4; j < 8; ++j) {
        out_scales[j] = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        out_mins[j]   = (uint8_t)((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4));
    }
}

int kiln_q4k_dequant_block(const uint8_t* src, float* dst) {
    if (src == NULL || dst == NULL) return 1;

    uint16_t d_h    = read_u16_le(src + 0);
    uint16_t dmin_h = read_u16_le(src + 2);
    float d    = f16_to_f32(d_h);
    float dmin = f16_to_f32(dmin_h);

    uint8_t scales[8];
    uint8_t mins[8];
    unpack_scales(src + 4, scales, mins);

    const uint8_t* qs = src + 16; /* 2 + 2 + 12 = 16 */

    for (int g = 0; g < 4; ++g) {
        const uint8_t* grp = qs + g * 32;
        float sc1 = d    * (float)scales[2 * g];
        float mn1 = dmin * (float)mins[2 * g];
        float sc2 = d    * (float)scales[2 * g + 1];
        float mn2 = dmin * (float)mins[2 * g + 1];
        for (int l = 0; l < 32; ++l) {
            uint8_t byte = grp[l];
            dst[g * 64 + l]      = sc1 * (float)(byte & 0x0F) - mn1;
            dst[g * 64 + 32 + l] = sc2 * (float)(byte >> 4)   - mn2;
        }
    }
    return 0;
}

int kiln_q4k_matmul_scalar(
    const uint8_t* weights,
    size_t num_weights,
    const float* activations,
    size_t num_activations,
    float* output)
{
    if (weights == NULL || activations == NULL || output == NULL) return 1;
    if (num_weights == 0) {
        *output = 0.0f;
        return 0;
    }
    if (num_activations < num_weights) return 2;

    size_t num_blocks = (num_weights + 255) / 256;
    float acc = 0.0f;

    float block[256];
    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* src = weights + b * 144;
        int rc = kiln_q4k_dequant_block(src, block);
        if (rc != 0) return rc;
        size_t base = b * 256;
        size_t count = (base + 256 <= num_weights) ? 256 : (num_weights - base);
        for (size_t i = 0; i < count; ++i) {
            acc += block[i] * activations[base + i];
        }
    }
    *output = acc;
    return 0;
}
