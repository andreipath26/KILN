# KILN Tokenizer Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for the text tokenizer. Anchors the BPE
implementation that turns text into tokens and back.

## Non-goals

This document does not specify SentencePiece. It does not specify
multilingual handling beyond UTF-8. It does not specify special token
injection for chat templates. Those are later units. This document
specifies byte-pair encoding (BPE) for the models KILN targets in
Phase 1.

## Why BPE first

The majority of modern open models use BPE: Llama 3, Qwen 2.5, Qwen 3,
Mistral, Gemma, and most of the fine-tunes on HuggingFace. GGUF stores
the BPE vocabulary and merge rules in the metadata. Implementing BPE
from scratch is about 200 lines of Rust and requires no external
dependency.

SentencePiece uses a unigram language model rather than BPE merges. It
is more complex. It is deferred to Phase 1 late or Phase 2.

## Where the tokenizer lives

The tokenizer lives in kiln-models, module tokenizer.rs. It uses the
GGUF metadata that the loader already parses. No new crate.

  pub struct BpeTokenizer {
      vocab: HashMap<String, u32>,
      merges: HashMap<(String, String), u32>,
      special_tokens: HashMap<String, u32>,
      inverse_vocab: Vec<String>,
  }

  impl BpeTokenizer {
      pub fn from_gguf(file: &GgufFile) -> Result<Self, TokenizerError>;
      pub fn encode(&self, text: &str) -> Vec<u32>;
      pub fn decode(&self, tokens: &[u32]) -> String;
      pub fn token_to_str(&self, token: u32) -> Option<&str>;
      pub fn str_to_token(&self, s: &str) -> Option<u32>;
  }

## The GGUF metadata keys

GGUF stores BPE data under these metadata keys:

  tokenizer.ggml.model            "gpt2" or "llama" for BPE
  tokenizer.ggml.tokens           array of strings, the vocabulary
  tokenizer.ggml.merges           array of strings, "a b" per merge rule
  tokenizer.ggml.bos_token_id     u32, beginning of sequence
  tokenizer.ggml.eos_token_id     u32, end of sequence
  tokenizer.ggml.padding_token_id u32, optional
  tokenizer.ggml.add_bos_token    bool, whether to prepend BOS
  tokenizer.ggml.add_eos_token    bool, whether to append EOS

The loader already parses arrays of strings and u32 values. The
tokenizer reads them from the GgufFile struct.

## The encode algorithm

Byte-pair encoding works in three steps.

Step 1. Split the input into UTF-8 characters. Each character becomes
its own token if it exists in the vocabulary.

Step 2. Iteratively merge adjacent token pairs. At each step, find the
pair that appears earliest in the merge rules. Merge it into a single
token. Repeat until no pair can be merged.

Step 3. Map each remaining token to its ID in the vocabulary.

A naive implementation is O(n^2) per merge and O(n^3) overall. For
short inputs (under 1000 characters), that is fine. For long inputs,
the implementation uses a priority queue keyed by merge rank. That is
O(n log n).

## The decode algorithm

Decoding is simpler. For each token ID, look up the string in the
vocabulary. Concatenate the strings. That is the output.

Special tokens (BOS, EOS, PAD) are decoded as empty strings unless
explicitly requested.

## Special tokens

Special tokens are entries in the vocabulary with a specific format.
The BPE model in GGUF uses angle brackets: <|endoftext|>, <|im_start|>,
<|im_end|>, and so on. The tokenizer treats these as opaque strings.
They do not merge with surrounding text.

The tokenizer exposes:

  pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32>;

This is what the chat loop calls. The chat loop passes add_bos and
add_eos based on the model's metadata.

## Error handling

  pub enum TokenizerError {
      MissingMetadata(String),
      MalformedMerge(String),
      UnknownToken(u32),
      InvalidUtf8,
  }

Each error names the specific key or token that failed. Rule KILN-E4
applied to the tokenizer boundary.

## Testing strategy

Four tests:

  1. Round trip. Encode a known string, decode it, verify equality.
     Test with empty string, single character, multi-word sentence,
     Unicode (emoji, CJK, accents).

  2. Known encoding. Encode a specific string and verify the token IDs
     match the expected values from the model's reference tokenizer.

  3. Special tokens. Encode with add_bos and add_eos. Verify the first
     and last tokens are the correct special IDs.

  4. Long input. Encode a 10,000-character text and verify it completes
     in under 100 milliseconds. That is the performance floor.

## What comes after the tokenizer

The chat loop. It calls the tokenizer to encode the user input, runs
the model to generate tokens, decodes each token back to a string, and
streams the output. The chat loop is a separate design document.

## Open questions

  - How does KILN handle models whose GGUF does not include the
    tokenizer metadata? Some old GGUF files ship without it.
  - How does KILN handle the byte-level fallback? GPT-2 BPE uses
    a byte-to-unicode mapping. Does KILN need to replicate it exactly?
  - What is the performance target for encode on a 4096-token prompt?
    Under 50 milliseconds is the working assumption.
  - Should KILN cache the merge rules as a sorted list for the priority
    queue implementation, or as a HashMap for the naive one?

## Version history

2026-10-03. Initial draft. BPE only. Reads from GGUF metadata. Encode
and decode algorithms. Special tokens. Error handling. Testing strategy.
Four open questions.
