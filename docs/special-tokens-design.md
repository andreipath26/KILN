# Special Tokens — Design

## The bug

`kiln tokenize` on the string
`<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n...`
produces byte-level BPE fragments:

  <   |   im   _start   |   >   system   \n   ...

It should produce the single control token ID 151644 for `<|im_start|>`
and 151645 for `<|im_end|>`. These IDs exist in the vocabulary. They are
marked as control tokens in `tokenizer.ggml.token_type`. The tokenizer
does not read that field, so it does not know to match them atomically.

Consequence: Qwen2.5-Instruct sees the wrong input during chat. The
ChatML structure is invisible to the model. Output quality is wrong.

## The fix

Load the special-token set from GGUF metadata. Before running BPE,
scan the input for any special-token substring. Split the input at
each match. BPE-encode the ordinary chunks. Emit the special token's
ID verbatim.

## Which tokens count as special

From `tokenizer.ggml.token_type`:

  type 1 = NORMAL
  type 2 = UNKNOWN
  type 3 = CONTROL
  type 4 = USER_DEFINED
  type 5 = UNUSED
  type 6 = BYTE

Specials to load atomically: types 3 (CONTROL) and 4 (USER_DEFINED).
Types 1, 2, 5, 6 go through normal BPE.

For Qwen2.5-1.5B that gives roughly 22 specials, including
<|endoftext|>, <|im_start|>, <|im_end|>, <|object_ref_start|>,
<|object_ref_end|>, <|box_start|>, <|box_end|>, <|quad_start|>,
<|quad_end|>, <|vision_start|>, <|vision_end|>, <|vision_pad|>,
<|image_pad|>, <|video_pad|>, and the tool-call tags.

## Data model

`BpeTokenizer` gains:

  special_tokens: Vec<(String, u32)>

Sorted longest-string-first so that overlapping prefixes match the
longer token, e.g. `<|im_start|>` before `<|im_`.

## Algorithm

encode(text):
  1. If special_tokens is empty, fall through to existing BPE.
  2. Scan text left to right. At each position, try each special
     in order (longest first). On a match:
       a. BPE-encode the ordinary chunk before the match.
       b. Push the special's ID.
       c. Advance past the match.
  3. BPE-encode the trailing chunk.
  4. Return the concatenation.

Complexity is O(len(text) * num_specials * avg_special_len). Fine for
chat prompts.

## Where the specials come from

`BpeTokenizer::from_gguf` reads two new keys:

  tokenizer.ggml.tokens     (already read)
  tokenizer.ggml.token_type (new)

Zips the two arrays. For every index where token_type is 3 or 4,
records (tokens[i], i as u32).

## EOS

The GGUF's eos_token_id is 151645, which is `<|im_end|>`. Correct for
ChatML. No change needed to ChatSession. Once the encoder emits 151645
for `<|im_end|>`, the session stops on it as it should.

## Tests

Test 1 — single special.
  encode("<|im_start|>") == [151644]

Test 2 — embedded.
  encode("a<|im_end|>b") tokenizes "a", then 151645, then "b",
  in that order, with no byte-level fragments.

Test 3 — the full ChatML wrapper.
  encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
  contains 151644, then the tokens for "user\nhi", then 151645, then
  151644, then the tokens for "assistant\n".

Test 4 — differential against llama.cpp.
  Use llama.cpp's tokenize on the same string and the same GGUF.
  Assert the ID lists are equal. This is the anchor that closes the
  whole class of tokenizer bugs.

## Anti-patterns

- Do not add specials to the merge table. Merges are for BPE.
- Do not match specials case-insensitively. GGUF specials are exact.
- Do not match shorter specials before longer ones. Longest first.
- Do not silently drop unknown specials. If a special is not in the
  vocabulary, it should not be in the special_tokens list.
