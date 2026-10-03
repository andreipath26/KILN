# KILN GGUF Loader Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for real model loading. Anchors the partial
GGUF loader that replaces the synthetic format for real inference.

## Non-goals

This document does not specify tokenization. It does not specify the
chat loop. It does not specify every quantization format. It specifies
a partial GGUF loader that reads real model files and supports the
tensor types KILN actually uses. Real tokenization is a separate
document.

## Why GGUF

GGUF is the de facto standard for local LLM distribution. It is the
format used by llama.cpp, Ollama, LM Studio, and the HuggingFace
ecosystem. Every model on HuggingFace that targets local inference
ships as GGUF. Choosing anything else would cut KILN off from the
entire model ecosystem.

GGUF is a binary container format. It has a text-like header (magic
number, version, counts), a metadata key-value table, a tensor info
table, and a data section. The metadata uses nine value types. The
tensors use many quantization formats.

## What KILN supports

KILN implements a partial GGUF loader. It reads:

  - The header: magic, version, tensor count, metadata count.
  - The metadata table: all nine GGUF value types.
  - The tensor info table: name, shape, type, offset.
  - The data section: raw bytes at the right offsets.

KILN supports these tensor types:

  - F32: 32-bit float. Used for activations and norms.
  - TQ1_0: KILN's own ternary format.
  - F16: 16-bit float. Used for some norms and embeddings.

Any other tensor type is rejected with a clear error naming the type
and the tensor. This is honest. A partial loader that refuses unknown
types is better than a loader that silently misinterprets data.

## Why TQ1_0 is a KILN type

TQ1_0 is not a standard GGUF type. It is KILN's 5-trits-per-byte
ternary packing from docs/kernels-design.md. To use it inside GGUF,
KILN uses a specific GGUF tensor type code (a value in the reserved
range) and stores the packed bytes as the tensor data. Models are
converted from standard GGUF to KILN GGUF by a separate tool.

This is an extension, not a fork. Any standard GGUF reader will see
the TQ1_0 tensor as an unknown type and skip it. KILN reads it. The
container format itself is unchanged.

## The loader interface

The loader lives in kiln-models, module gguf.rs.

  pub struct GgufFile {
      pub version: u32,
      pub metadata: HashMap<String, GgufValue>,
      pub tensors: Vec<GgufTensorInfo>,
      pub data_offset: u64,
      pub mmap: Option<Mmap>,
  }

  pub struct GgufTensorInfo {
      pub name: String,
      pub shape: Vec<u64>,
      pub dtype: GgufType,
      pub offset: u64,
      pub byte_size: u64,
  }

  pub enum GgufType {
      F32,
      F16,
      Tq1_0,
      Unknown(u32),
  }

  pub fn read_gguf(path: &Path) -> Result<GgufFile, GgufError>;

  impl GgufFile {
      pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]>;
      pub fn metadata_str(&self, key: &str) -> Option<&str>;
      pub fn metadata_u32(&self, key: &str) -> Option<u32>;
  }

The loader uses memory-mapped file access. The file is mapped once.
Reading a tensor is a slice into the mapped region. No copying.

## The metadata types

GGUF metadata values have one of nine types:

  0  UINT8
  1  INT8
  2  UINT16
  3  INT16
  4  UINT32
  5  INT32
  6  FLOAT32
  7  BOOL
  8  STRING
  9  ARRAY

The loader represents them as a Rust enum:

  pub enum GgufValue {
      U8(u8),
      I8(i8),
      U16(u16),
      I16(i16),
      U32(u32),
      I32(i32),
      F32(f32),
      Bool(bool),
      String(String),
      Array(Vec<GgufValue>),
  }

The array element type is encoded in the GGUF file as a separate
value before the array length. The loader reads it and enforces that
every element has that type.

## The alignment rule

GGUF aligns the data section to a boundary specified in the metadata
key general.alignment. The default is 32 bytes. The loader reads the
alignment, rounds the offset up to the nearest multiple, and begins
the data section there.

Getting this wrong means every tensor reads the wrong bytes. The
loader computes the offset as:

  data_offset = align_up(position_after_tensor_table, alignment)

The align_up function rounds up to the nearest multiple of alignment.

## Little-endian only

GGUF is little-endian. All integer and float reads use little-endian
byte order. On a big-endian machine, the loader must byte-swap. KILN
targets little-endian hardware only. Big-endian support is a Phase 3
or later concern.

## Error handling

The loader returns a GgufError for:

  - File not found or unreadable.
  - Bad magic number.
  - Unsupported version (KILN supports version 3).
  - Truncated metadata or tensor table.
  - Metadata value type out of range.
  - Alignment value is not a power of two.
  - Tensor offset is past the end of the file.
  - Tensor byte size does not match the shape and dtype.

Each error names the exact byte position and the exact reason. Rule
KILN-E4 applied to the loader boundary.

## What KILN does not implement in Phase 1

Tokenization is separate. The loader returns raw bytes. Turning bytes
into tokens is the tokenizer's job.

The chat template is separate. GGUF stores the chat template in the
metadata under tokenizer.chat_template. The loader exposes it as a
string. Applying it is the chat layer's job.

Tensor arithmetic is separate. The loader returns raw bytes. The
fused matmul kernel consumes them. Nothing in between.

## Testing strategy

Four tests:

  1. Round-trip on a small synthetic GGUF. Write a GGUF in memory,
     read it back, verify the metadata and tensors match.
  2. Header validation. Feed the loader a file with a bad magic
     number and a version 99 header. Both must be rejected.
  3. Truncation. Cut a valid GGUF file in half. The loader must reject
     it with a clear error.
  4. Real model. Load a small real GGUF from HuggingFace, verify the
     metadata and the tensor list.

The fourth test requires network access on the first run. It is marked
ignore by default. It runs only when explicitly requested.

## The conversion tool

A separate binary, kiln-convert, reads a standard GGUF and writes a
KILN GGUF with TQ1_0 ternary weights. It takes a standard model and
runs the ternarization pipeline. The conversion tool is Phase 1 late
or Phase 2.

For Phase 1, KILN loads a KILN GGUF. To test the loader before the
converter exists, we write a small synthetic KILN GGUF in the test.

## Integration with the pipeline

The pipeline currently reads a synthetic model. After GGUF loading
lands, it reads a real GGUF and runs the same five steps. The pipeline
function gains one parameter: a flag for which loader to use.

  pub fn run_once(path: PathBuf, loader: Loader) -> Result<PipelineReport, PipelineError>;

  pub enum Loader {
      Synthetic,
      Gguf,
  }

The pipeline does not care which format the model is in. It cares that
the loader returns packed TQ1_0 weights and int8 activations. Both
formats provide those.

## Open questions

  - What GGUF version does KILN target? Version 3 is the current
    standard. Version 2 is deprecated but still common. Should the
    loader support both?
  - How does the loader handle files larger than the address space?
    On a 32-bit system, a 10 GB GGUF cannot be mapped. The loader
    needs a fallback to seeking.
  - What is the exact tensor type code for TQ1_0? It must be in the
    reserved range to avoid conflicts with future GGUF types. Which
    value?
  - How does the loader handle the special token IDs? They are stored
    as metadata keys with names like tokenizer.ggml.bos_token_id.
    These are exposed as values but not interpreted in Phase 1.

## Version history

2026-10-03. Initial draft. Partial GGUF loader, supported tensor types,
TQ1_0 extension, loader interface, metadata types, alignment rule,
little-endian constraint, error handling, testing strategy, conversion
tool, pipeline integration, four open questions.
