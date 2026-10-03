/* KILN kernel: Q4_K dequantization and fused matmul.
 *
 * See docs/q4k-design.md for the format specification.
 *
 * Q4_K super-block: 256 weights, 144 bytes.
 *   2 bytes: d    (F16 super-block scale)
 *   2 bytes: dmin (F16 super-block minimum)
 *  12 bytes: 8 sub-block scales and 8 sub-block mins, 6-bit each
 * 128 bytes: 256 4-bit weights, 2 per byte
 */

#ifndef KILN_Q4K_H
#define KILN_Q4K_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantize one Q4_K super-block into 256 float values.
 *
 * src must point to 144 bytes of Q4_K data.
 * dst must have room for 256 floats.
 *
 * Returns 0 on success.
 */
int kiln_q4k_dequant_block(const uint8_t* src, float* dst);

/* Fused matmul: dot product of a Q4_K weight row and an f32 activation
 * vector. The weight row has num_weights trits packed as Q4_K.
 *
 * Returns 0 on success.
 */
int kiln_q4k_matmul_scalar(
    const uint8_t* weights,
    size_t num_weights,
    const float* activations,
    size_t num_activations,
    float* output
);

#ifdef __cplusplus
}
#endif

#endif /* KILN_Q4K_H */
