/* KILN llama.cpp shim implementation.
 * Includes llama.h and uses the real structs by value. Rust never
 * sees them. See docs/runtime-loading-design.md.
 */

#include "kiln_llama_shim.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>

struct kiln_llama {
    struct llama_model*   model;
    struct llama_context* ctx;
    const struct llama_vocab* vocab;
    int32_t n_vocab;
    int32_t n_past;
    struct llama_batch batch;
};

kiln_llama_t kiln_llama_load(const char* model_path) {
    if (!model_path) return NULL;

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    /* 2048 token context is enough for Phase 2. */
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return NULL;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct kiln_llama* k = (struct kiln_llama*)calloc(1, sizeof(struct kiln_llama));
    if (!k) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    k->model   = model;
    k->ctx     = ctx;
    k->vocab   = vocab;
    k->n_vocab = n_vocab;
    k->n_past  = 0;
    k->batch   = llama_batch_init(512, 0, 1);
    return (kiln_llama_t)k;
}

void kiln_llama_free(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return;
    /* Order matters. Free the batch before the context, free the
     * context before the model. Do not call llama_backend_free here:
     * it is process-global and other contexts may still exist. */
    llama_batch_free(k->batch);
    llama_free(k->ctx);
    llama_model_free(k->model);
    free(k);
}

int32_t kiln_llama_n_vocab(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    return k ? k->n_vocab : 0;
}

int32_t kiln_llama_tokenize(
    kiln_llama_t h, const char* text, int32_t text_len,
    int32_t* out_tokens, int32_t out_cap,
    int add_special, int parse_special)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    return llama_tokenize(
        k->vocab, text, text_len, out_tokens, out_cap,
        add_special != 0, parse_special != 0);
}

int32_t kiln_llama_decode(
    kiln_llama_t h, const int32_t* tokens, int32_t n_tokens, int32_t pos_offset)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;

    /* Single-sequence decode. Every token belongs to sequence 0.
     * llama_batch_init already allocated seq_id[] as an array of
     * pointers, each pointing at a heap array of size n_seq_max.
     * Write into those arrays. Do not replace the pointers, or
     * llama_batch_free will try to free a stack address. */
    k->batch.n_tokens = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        k->batch.token[i]       = tokens[i];
        k->batch.pos[i]         = pos_offset + i;
        k->batch.n_seq_id[i]    = 1;
        k->batch.seq_id[i][0]   = 0;
        k->batch.logits[i]      = (i == n_tokens - 1) ? 1 : 0;
    }
    k->batch.n_tokens = n_tokens;

    int32_t rc = llama_decode(k->ctx, k->batch);
    if (rc == 0) k->n_past = pos_offset + n_tokens;
    return rc;
}

int32_t kiln_llama_logits(kiln_llama_t h, float* out, int32_t out_cap) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    float* p = llama_get_logits(k->ctx);
    if (!p) return -2;
    int32_t n = k->n_vocab < out_cap ? k->n_vocab : out_cap;
    memcpy(out, p, (size_t)n * sizeof(float));
    return n;
}

int32_t kiln_llama_token_to_str(
    kiln_llama_t h, int32_t token, char* buf, int32_t buf_len)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    return llama_token_to_piece(k->vocab, token, buf, buf_len, 0, 1);
}
