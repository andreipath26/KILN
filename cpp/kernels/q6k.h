/* KILN kernel: Q6_K dequantization and fused matmul.
 *
 * Q6_K super-block: 256 weights, 210 bytes.
 *  128 bytes: ql, the low 4 bits of each 6-bit weight
 *   64 bytes: qh, the high 2 bits of each 6-bit weight
 *   16 bytes: int8 scales, one per 16-weight sub-block
 *    2 bytes: d, the super-block scale (F16)
 */

#ifndef KILN_Q6K_H
#define KILN_Q6K_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int kiln_q6k_dequant_block(const uint8_t* src, float* dst);

int kiln_q6k_matmul_scalar(
    const uint8_t* weights,
    size_t num_weights,
    const float* activations,
    size_t num_activations,
    float* output
);

#ifdef __cplusplus
}
#endif

#endif /* KILN_Q6K_H */
