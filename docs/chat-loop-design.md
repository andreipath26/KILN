# KILN Chat Loop Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for the generation loop. Anchors the piece
that turns a single matmul into a conversation.

## Non-goals

This document does not specify the transformer forward pass. It does
not specify attention, layer normalization, or residual connections.
Those are separate units. This document specifies the generation loop
that wraps them: encode input, sample tokens, decode output, stream.

## Why this exists

The pipeline runs one matmul and returns a single number. A chat loop
runs hundreds of matmuls in sequence and returns text. The gap between
them is the generation loop.

For Phase 1, the chat loop runs against the synthetic pipeline. The
"model" produces one number per forward pass. To make a conversation,
we need a real forward pass that produces a logits vector with one
value per vocabulary token.

The chat loop design is honest about this: it defines the shape of the
generation loop and works with whatever forward pass we have. When the
real transformer lands, the loop stays the same.

## The generation loop

Five steps per token.

Step 1. Encode. Convert the user's text to token IDs with the
tokenizer. Prepend BOS if the model requests it.

Step 2. Forward. Run the token sequence through the model. Get the
logits for the next token.

Step 3. Sample. Pick one token ID from the logits according to the
sampling strategy. Append it to the sequence.

Step 4. Decode. Convert the new token ID back to text with the
tokenizer.

Step 5. Stream. Print the text as it is produced.

Repeat from step 2 until the model emits EOS or until the token limit
is reached.

## The forward pass interface

The chat loop does not care how the forward pass works. It calls one
function:

  pub trait Forward {
      /// Given a sequence of token IDs, return the logits for the
      /// next token. The logits vector has one value per vocabulary
      /// entry.
      fn forward(&mut self, tokens: &[u32]) -> Vec<f32>;
  }

Implementations of Forward:
  - RealTransformer: the actual model. Phase 1 late.
  - SyntheticForward: a placeholder that produces pseudo-random
    logits from a seed. Phase 1 for testing the loop.
  - MockForward: returns fixed logits. For unit tests.

## The sampling strategy

Three strategies, selected by a config struct.

  pub enum SamplingStrategy {
      /// Always pick the highest-probability token.
      Greedy,
      /// Sample from the top-k most likely tokens.
      TopK { k: usize, temperature: f32 },
      /// Sample from tokens whose cumulative probability exceeds a
      /// threshold.
      TopP { p: f32, temperature: f32 },
  }

Greedy is deterministic and is the default for tests. TopK and TopP
are for real use. The temperature scales the logits before sampling.

The sampler takes a logits vector and a random seed. Given the same
logits and the same seed, it produces the same token. That is required
by AT-7.

## The random number generator

For determinism, the sampler uses a seeded PRNG. We do not depend on
the OS random source. We use a small xorshift or PCG generator that
is easy to seed and reproduce.

  pub struct Sampler {
      seed: u64,
      state: u64,
  }

The seed is part of the chat session. Two sessions with the same seed
and the same inputs produce the same output.

## The chat session

A chat session holds the tokenizer, the forward pass, the sampler,
and the conversation history.

  pub struct ChatSession {
      pub tokenizer: BpeTokenizer,
      pub forward: Box<dyn Forward>,
      pub sampler: Sampler,
      pub history: Vec<u32>,
      pub max_new_tokens: usize,
      pub eos_token_id: Option<u32>,
  }

  impl ChatSession {
      pub fn new(...) -> Self;
      pub fn generate(&mut self, prompt: &str) -> String;
      pub fn generate_streaming(&mut self, prompt: &str, sink: &mut dyn FnMut(&str)) -> String;
  }

The `generate` method returns the full response after it is complete.
The `generate_streaming` method calls the sink for each new token.

## Streaming output

The streaming interface is a closure. The caller provides a function
that takes a `&str` and does something with it. The chat session calls
the sink after every decoded token.

  fn print_to_stdout(s: &str) {
      print!("{}", s);
      std::io::stdout().flush().unwrap();
  }

The CLI calls `generate_streaming(prompt, &mut print_to_stdout)`. That
is what makes the output appear token by token.

## Errors

  pub enum ChatError {
      Tokenizer(TokenizerError),
      Forward(String),
      EmptyVocabulary,
      NoEosToken,
  }

Each error names the step that failed and the reason.

## Testing strategy

Three tests:

  1. Determinism. Run a chat session twice with the same seed and the
     same prompt. Verify the outputs are byte-identical.

  2. Length. Run a session with max_new_tokens = 10. Verify the output
     has at most 10 tokens.

  3. EOS. Run a session with a mock forward that emits EOS on the
     third token. Verify the loop stops at three tokens.

The tests use a MockForward with deterministic output so they do not
depend on a real model.

## What comes after the chat loop

The real transformer forward pass. It replaces SyntheticForward with
a real implementation that reads weights from the GGUF loader and
runs attention, layer norm, and the feed-forward network. That is a
separate design document.

## Open questions

  - Does the chat loop accumulate the KV cache across forward passes,
    or does it re-run the full sequence every time? Re-running is
    simpler but slower. The KV cache is a Phase 2 concern.
  - How does the chat loop handle multi-turn conversations? The
    history grows. When it exceeds the context window, what gets
    dropped?
  - How does the chat loop report token-by-token latency? The user
    wants to see time-to-first-token and tokens-per-second.
  - Should the sampler be part of the chat session or separate? The
    current design puts it inside. If it needs to be swapped mid-session,
    that becomes a method call.

## Version history

2026-10-03. Initial draft. Five-step generation loop. Forward trait
with three implementations. Three sampling strategies. Seeded PRNG
for determinism. ChatSession struct. Streaming via closure. Three
tests. Four open questions.
