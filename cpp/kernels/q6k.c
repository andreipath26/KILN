/* KILN kernel: Q6_K dequantization and fused matmul.
 *
 * The layout matches ggml's dequantize_row_q6_K in ggml-quants.c
 * exactly. This is the authoritative reference.
 *
 * Q6_K super-block: 256 weights, 210 bytes.
 *  128 bytes: ql
 *   64 bytes: qh
 *   16 bytes: int8 scales
 *    2 bytes: d (F16)
 *
 * The dequantization loops over two 128-weight halves. Each half
 * processes 32 output positions, writing 4 values per iteration.
 */

#include "q6k.h"
#include <string.h>

static float f16_to_f32_q6k(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= 0x3FFu;
    } else if (exp == 0x1Fu) {
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

static uint16_t read_u16_le_q6k(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

int kiln_q6k_dequant_block(const uint8_t* src, float* dst) {
    if (src == NULL || dst == NULL) return 1;

    const uint8_t* ql = src;                        /* 128 bytes */
    const uint8_t* qh = src + 128;                  /* 64 bytes */
    const int8_t*  sc = (const int8_t*)(src + 192); /* 16 bytes */
    uint16_t d_h = read_u16_le_q6k(src + 208);
    float d = f16_to_f32_q6k(d_h);

    /* The outer loop runs twice, each covering 128 output weights.
     * Inside, 32 iterations each write 4 values. */
    float* y = dst;
    const uint8_t* ql_cur = ql;
    const uint8_t* qh_cur = qh;
    const int8_t*  sc_cur = sc;

    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = (int)((ql_cur[l +  0] & 0xF) | (((qh_cur[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql_cur[l + 32] & 0xF) | (((qh_cur[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql_cur[l +  0] >>  4) | (((qh_cur[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql_cur[l + 32] >>  4) | (((qh_cur[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * (float)sc_cur[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc_cur[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc_cur[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc_cur[is + 6] * (float)q4;
        }
        y     += 128;
        ql_cur += 64;
        qh_cur += 32;
        sc_cur += 8;
    }

    return 0;
}

int kiln_q6k_matmul_scalar(
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
        const uint8_t* src = weights + b * 210;
        int rc = kiln_q6k_dequant_block(src, block);
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
