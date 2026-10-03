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

#ifdef __cplusplus
}
#endif
