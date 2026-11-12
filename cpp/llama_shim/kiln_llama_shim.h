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

/* Context parameters. Filled by the caller from the system profile. */
struct kiln_llama_params {
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_threads;
    /* Opt into on-demand mmap reads for supported routed-expert tensors. */
    int32_t lazy_moe_experts;
};

/* Returns NULL on failure. */
kiln_llama_t kiln_llama_load(const char* model_path, const struct kiln_llama_params* params);

void kiln_llama_free(kiln_llama_t h);

int32_t kiln_llama_n_vocab(kiln_llama_t h);

/* LLaDA helpers for the opt-in Phase 6 diffusion path. */
int32_t kiln_llama_mask_token(kiln_llama_t h);
/* Returns 1 when model metadata/default shifts diffusion logits, otherwise 0. */
int32_t kiln_llama_shift_logits(kiln_llama_t h);
/* Apply the model's native chat template to a single user prompt.
 * Returns required/written bytes excluding NUL, negative on unsupported/error. */
int32_t kiln_llama_apply_user_template(
    kiln_llama_t h, const char* prompt, int32_t prompt_len,
    char* out, int32_t out_cap);
/* Run one full non-causal denoising forward pass and return top-1 candidates,
 * their probabilities, and the captured [query][key] CLAD attention matrix. */
int32_t kiln_llama_diffusion_step(
    kiln_llama_t h,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t* predicted_tokens,
    float* confidence,
    float* attention,
    size_t attention_capacity);

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
    int32_t pos_offset,
    int32_t request_last_logits);

/* Copy n_vocab logits for the last decoded position into out. */
int32_t kiln_llama_logits(kiln_llama_t h, float* out, int32_t out_cap);

/* Return greedy argmax from the last logits without copying the vocabulary.
 * Matches the stable first-max behavior of the Rust greedy sampler.
 * Negative values indicate an invalid handle or missing logits. */
int32_t kiln_llama_greedy_token(kiln_llama_t h);

/* Verify n proposed greedy tokens in a single llama_decode batch.
 * Returns accepted draft count (0..n), and writes the corrected next
 * target token. Invalid state or unsupported KV rollback returns -1.
 * After success only accepted draft tokens remain in the KV cache. */
int32_t kiln_llama_verify_greedy_draft(
    kiln_llama_t h, const int32_t* draft, int32_t n,
    int32_t* next_token);


/* Decode one token to a UTF-8 string into buf. Returns bytes written. */
int32_t kiln_llama_token_to_str(
    kiln_llama_t h,
    int32_t token,
    char* buf,
    int32_t buf_len);

/* Reset the KV cache and the position counter. Use between turns of a
 * multi-turn conversation when the context is being reused for a new
 * session. Returns 0 on success. */
int32_t kiln_llama_reset(kiln_llama_t h);

/* Strictly bounded in-memory KV snapshot interface.
 * These opaque bytes are valid only for the same model/context parameters.
 * KILN's Rust wrapper enforces same-context identity and byte caps. */
size_t kiln_llama_state_size(kiln_llama_t h);
size_t kiln_llama_state_get(kiln_llama_t h, uint8_t* dst, size_t capacity);
/* Return 0 only if the full buffer restores successfully; reset on failure. */
int32_t kiln_llama_state_set(
    kiln_llama_t h, const uint8_t* src, size_t length, int32_t n_past);

/* Return the EOS token id for this model. -1 if the model has no EOS. */
int32_t kiln_llama_eos_token(kiln_llama_t h);

#ifdef __cplusplus
}
#endif

#endif /* KILN_LLAMA_SHIM_H */
