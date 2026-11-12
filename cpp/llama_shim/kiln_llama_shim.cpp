/* KILN llama.cpp shim implementation.
 * Includes llama.h and uses the real structs by value. Rust never
 * sees them. See docs/runtime-loading-design.md.
 */

#include "kiln_llama_shim.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <string>

struct kiln_llama {
    struct llama_model*   model;
    struct llama_context* ctx;
    const struct llama_vocab* vocab;
    int32_t n_vocab;
    int32_t n_past;
    int32_t batch_capacity;
    bool shift_logits;
    struct llama_batch batch;
};

kiln_llama_t kiln_llama_load(const char* model_path, const struct kiln_llama_params* params) {
    if (!model_path) return NULL;

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    if (params && params->lazy_moe_experts) {
        mparams.lazy_mode = LLAMA_LAZY_MODE_ON;
    }
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    if (llama_model_is_diffusion(model)) {
        // CLAD requires materialized attention probabilities rather than a fused kernel.
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (params) {
        if (params->n_ctx > 0)     cparams.n_ctx = (uint32_t)params->n_ctx;
        if (params->n_batch > 0)   cparams.n_batch = (uint32_t)params->n_batch;
        if (params->n_ubatch > 0)  cparams.n_ubatch = (uint32_t)params->n_ubatch;
        if (params->n_threads > 0) {
            cparams.n_threads = (int32_t)params->n_threads;
            // llama.cpp distinguishes single-token decode from multi-token
            // prefill. Before this, --threads changed decode threads only:
            // prompt/batch execution kept llama.cpp's independent default.
            cparams.n_threads_batch = (int32_t)params->n_threads;
        }
    } else {
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
    }
    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return NULL;
    }
    if (llama_model_is_diffusion(model)) {
        // Masked diffusion requires every position to attend across the full sequence.
        llama_set_causal_attn(ctx, false);
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
    char shift_logits_value[8] = {};
    k->shift_logits =
        llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_value, sizeof(shift_logits_value)) < 0 ||
        strcmp(shift_logits_value, "true") == 0;
    k->n_past = 0;
    /* Keep the reusable batch at least as large as the configured
     * context batch, otherwise valid caller batches can be rejected
     * or overrun the allocated token arrays. */
    k->batch_capacity = (cparams.n_batch > 0 && cparams.n_batch <= (uint32_t)INT32_MAX)
        ? (int32_t)cparams.n_batch : 512;
    k->batch = llama_batch_init(k->batch_capacity, 0, 1);
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

int32_t kiln_llama_mask_token(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !llama_model_is_diffusion(k->model)) return -1;
    return llama_vocab_mask(k->vocab);
}

int32_t kiln_llama_shift_logits(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    return k && llama_model_is_diffusion(k->model) ? (k->shift_logits ? 1 : 0) : -1;
}

int32_t kiln_llama_apply_user_template(
    kiln_llama_t h, const char* prompt, int32_t prompt_len,
    char* out, int32_t out_cap)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !prompt || prompt_len < 0 || !out || out_cap <= 0) return -1;
    const char* tmpl = llama_model_chat_template(k->model, nullptr);
    if (!tmpl) return -2;
    const std::string content(prompt, (size_t) prompt_len);
    const llama_chat_message message = {"user", content.c_str()};
    return llama_chat_apply_template(tmpl, &message, 1, true, out, out_cap);
}

int32_t kiln_llama_diffusion_step(
    kiln_llama_t h, const int32_t* tokens, int32_t n_tokens,
    int32_t capture_attention, int32_t* predicted_tokens, float* confidence,
    float* attention, size_t attention_capacity)
{
    struct kiln_llama* k = (struct kiln_llama*) h;
    if (!k || !tokens || !predicted_tokens || !confidence || (capture_attention && !attention) ||
        n_tokens <= 0 || n_tokens > k->batch_capacity ||
        !llama_model_is_diffusion(k->model)) {
        return -1;
    }
    const size_t token_count = (size_t) n_tokens;
    if (capture_attention &&
        (token_count > SIZE_MAX / token_count || token_count * token_count > attention_capacity)) {
        return -2;
    }

    // Keep the call self-contained if another operation has changed context mode.
    llama_set_causal_attn(k->ctx, false);
    llama_set_clad_attention_capture(k->ctx, capture_attention != 0);

    k->batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        k->batch.token[i] = tokens[i];
        k->batch.pos[i] = i;
        k->batch.n_seq_id[i] = 1;
        k->batch.seq_id[i][0] = 0;
        k->batch.logits[i] = 1;
    }
    const int32_t decode_rc = llama_decode(k->ctx, k->batch);
    if (decode_rc != 0) {
        k->n_past = 0;
        return -3;
    }

    for (int32_t pos = 0; pos < n_tokens; ++pos) {
        const int32_t logits_pos = k->shift_logits && pos > 0 ? pos - 1 : pos;
        const float* row = llama_get_logits_ith(k->ctx, logits_pos);
        if (!row) return -4;
        int32_t best = 0;
        float maximum = row[0];
        if (!std::isfinite(maximum)) return -5;
        for (int32_t token = 1; token < k->n_vocab; ++token) {
            if (!std::isfinite(row[token])) return -5;
            if (row[token] > maximum) {
                maximum = row[token];
                best = token;
            }
        }
        double sum = 0.0;
        for (int32_t token = 0; token < k->n_vocab; ++token) {
            sum += std::exp((double) row[token] - (double) maximum);
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) return -6;
        predicted_tokens[pos] = best;
        confidence[pos] = (float) (1.0 / sum);
    }

    k->n_past = 0;
    if (!capture_attention) {
        return 0;
    }

    const uint32_t attention_tokens = llama_get_clad_attention_tokens(k->ctx);
    if (attention_tokens != (uint32_t) n_tokens) {
        k->n_past = 0;
        return -7;
    }
    const size_t expected = token_count * token_count;
    const size_t copied = llama_get_clad_attention(k->ctx, attention, attention_capacity);
    k->n_past = 0;
    if (copied != expected) return -8;
    return 0;
}

int32_t kiln_llama_tokenize(
    kiln_llama_t h, const char* text, int32_t text_len,
    int32_t* out_tokens, int32_t out_cap,
    int add_special, int parse_special)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !text || text_len < 0 || !out_tokens || out_cap <= 0) return -1;
    return llama_tokenize(
        k->vocab, text, text_len, out_tokens, out_cap,
        add_special != 0, parse_special != 0);
}

int32_t kiln_llama_decode(
    kiln_llama_t h, const int32_t* tokens, int32_t n_tokens, int32_t pos_offset, int32_t request_last_logits)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !tokens || n_tokens <= 0 || n_tokens > k->batch_capacity || pos_offset < 0) return -1;

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
        k->batch.logits[i]      = (request_last_logits != 0 && i == n_tokens - 1) ? 1 : 0;
    }
    k->batch.n_tokens = n_tokens;

    int32_t rc = llama_decode(k->ctx, k->batch);
    if (rc == 0) k->n_past = pos_offset + n_tokens;
    return rc;
}

int32_t kiln_llama_logits(kiln_llama_t h, float* out, int32_t out_cap) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !out || out_cap <= 0) return -1;
    float* p = llama_get_logits(k->ctx);
    if (!p) return -2;
    int32_t n = k->n_vocab < out_cap ? k->n_vocab : out_cap;
    memcpy(out, p, (size_t)n * sizeof(float));
    return n;
}

int32_t kiln_llama_greedy_token(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || k->n_vocab <= 0) return -1;
    const float* logits = llama_get_logits(k->ctx);
    if (!logits) return -2;
    // Strictly greater preserves the first index for equal maxima and
    // matches the existing Rust Sampler::Greedy comparison order.
    int32_t best = 0;
    float best_value = logits[0];
    for (int32_t i = 1; i < k->n_vocab; ++i) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    return best;
}


/* First-max greedy argmax matches Rust's SamplingStrategy::Greedy. */
static int32_t kiln_greedy_argmax(const float* logits, int32_t vocab) {
    if (!logits || vocab <= 0) return -1;
    int32_t best = 0;
    float maximum = logits[0];
    for (int32_t i = 1; i < vocab; ++i) {
        if (logits[i] > maximum) {
            maximum = logits[i];
            best = i;
        }
    }
    return best;
}

int32_t kiln_llama_verify_greedy_draft(
    kiln_llama_t h, const int32_t* draft, int32_t n, int32_t* next_token)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !draft || !next_token || n <= 0 || n > k->batch_capacity) return -1;
    if (k->n_past < 0 || n > INT32_MAX - k->n_past) return -1;

    // Precondition: the preceding prefill/decode requested logits at its
    // last position. Verify the first proposal before mutating the KV cache.
    const int32_t first = kiln_greedy_argmax(llama_get_logits_ith(k->ctx, -1), k->n_vocab);
    if (first < 0) return -1;
    if (draft[0] != first) {
        *next_token = first;
        return 0;
    }

    const int32_t before = k->n_past;
    k->batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        k->batch.token[i] = draft[i];
        k->batch.pos[i] = before + i;
        k->batch.n_seq_id[i] = 1;
        k->batch.seq_id[i][0] = 0;
        // Every candidate boundary must be checked against its predecessor.
        k->batch.logits[i] = 1;
    }
    if (llama_decode(k->ctx, k->batch) != 0) return -1;
    k->n_past = before + n;

    // The first token is already verified from the pre-existing logits.
    for (int32_t i = 1; i < n; ++i) {
        const int32_t expected = kiln_greedy_argmax(
            llama_get_logits_ith(k->ctx, i - 1), k->n_vocab);
        if (expected < 0) return -1;
        if (draft[i] != expected) {
            // The KV cache must not retain rejected or unverified proposals.
            llama_memory_t mem = llama_get_memory(k->ctx);
            if (!mem || !llama_memory_seq_rm(mem, 0, before + i, -1)) return -1;
            k->n_past = before + i;
            *next_token = expected;
            return i;
        }
    }
    const int32_t bonus = kiln_greedy_argmax(
        llama_get_logits_ith(k->ctx, n - 1), k->n_vocab);
    if (bonus < 0) return -1;
    *next_token = bonus;
    return n;
}

int32_t kiln_llama_token_to_str(
    kiln_llama_t h, int32_t token, char* buf, int32_t buf_len)
{
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    return llama_token_to_piece(k->vocab, token, buf, buf_len, 0, 1);
}

int32_t kiln_llama_reset(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    /* v0.5.0 uses the memory API, not the deprecated kv_self API. */
    llama_memory_t mem = llama_get_memory(k->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    k->n_past = 0;
    return 0;
}

size_t kiln_llama_state_size(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    return k ? llama_state_get_size(k->ctx) : 0;
}

size_t kiln_llama_state_get(kiln_llama_t h, uint8_t* dst, size_t capacity) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !dst || capacity == 0) return 0;
    return llama_state_get_data(k->ctx, dst, capacity);
}

int32_t kiln_llama_state_set(
    kiln_llama_t h, const uint8_t* src, size_t length, int32_t n_past) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k || !src || length == 0 || n_past <= 0) return -1;
    const size_t restored = llama_state_set_data(k->ctx, src, length);
    if (restored != length) {
        // Do not allow a caller to continue from a partially modified KV
        // allocation after a corrupt or truncated snapshot.
        kiln_llama_reset(h);
        return -1;
    }
    k->n_past = n_past;
    return 0;
}

int32_t kiln_llama_eos_token(kiln_llama_t h) {
    struct kiln_llama* k = (struct kiln_llama*)h;
    if (!k) return -1;
    return llama_vocab_eos(k->vocab);
}
