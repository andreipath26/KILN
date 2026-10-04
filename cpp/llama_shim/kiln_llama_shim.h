/* KILN llama.cpp shim.
 * Opaque handle. Rust never sees llama.cpp structs.
 * Compile: see crates/kiln-runtime/build.rs.
 */
#ifndef KILN_LLAMA_SHIM_H
#define KILN_LLAMA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* kiln_llama_t;

/* Returns NULL on failure. */
kiln_llama_t kiln_llama_load(const char* model_path);

void kiln_llama_free(kiln_llama_t h);

int32_t kiln_llama_n_vocab(kiln_llama_t h);

/* Returns number of tokens written, or negative on error. */
int32_t kiln_llama_tokenize(
    kiln_llama_t h,
    const char* text,
    int32_t text_len,
    int32_t* out_tokens,
    int32_t out_cap,
    int add_special,
    int parse_special);

/* Run a forward pass on the given tokens. Returns 0 on success. */
int32_t kiln_llama_decode(
    kiln_llama_t h,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t pos_offset);

/* Copy n_vocab logits for the last decoded position into out. */
int32_t kiln_llama_logits(kiln_llama_t h, float* out, int32_t out_cap);

/* Decode one token to a UTF-8 string into buf. Returns bytes written. */
int32_t kiln_llama_token_to_str(
    kiln_llama_t h,
    int32_t token,
    char* buf,
    int32_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* KILN_LLAMA_SHIM_H */
