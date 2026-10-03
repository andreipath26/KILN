# KILN Kernel Layer Design

Status: DRAFT
Owner: project lead
Created: 2026-10-03
Purpose: Design document for the kernel layer. Anchors the TQ1.0 format,
the C ABI, the Rust FFI contract, and the build process.

## Non-goals

This document does not specify the AVX2 implementation. It does not specify
the ARM NEON implementation. It specifies the format, the ABI, and the build
integration. The actual SIMD code lands in the implementation.

## Why this layer exists

Every performance claim in the KILN roadmap depends on the kernels being
correct and fast. The TQ1.0 format is the on-disk representation of ternary
weights. Once a model is packed in this format and shipped, changing the
format means repacking every model. The ABI is the contract between the
Rust runtime and the C kernels. Once it is locked, changing it means
rebuilding every binding. This document locks both.

## The TQ1.0 Format

TQ1.0 is a ternary packing format. Each weight is a trit with one of three
values: -1, 0, +1. Five trits pack into one byte.

Why five trits per byte. Because 3^5 = 243 and 243 is less than 256. The
packed value is a Base-3 number where each digit is a trit shifted from
{0,1,2} to {-1,0,+1}.

Encoding:

  trit_to_digit(t) = t + 1   # -1 -> 0, 0 -> 1, +1 -> 2
  digit_to_trit(d) = d - 1

  packed = d0*81 + d1*27 + d2*9 + d3*3 + d4*1

where d0 is the first trit and d4 is the fifth. The result is in [0, 242].
Values 243 through 255 are invalid and reserved for future use.

Decoding:

  d4 = packed % 3
  d3 = (packed / 3) % 3
  d2 = (packed / 9) % 3
  d1 = (packed / 27) % 3
  d0 = (packed / 81) % 3

  trit_i = digit_to_trit(d_i)

Alignment and padding. If the input tensor length is not a multiple of five,
the last byte is padded with zero trits. The pad trits are ignored during
inference. The total byte count is ceil(n / 5).

Why TQ1.0 and not TQ1.58. BitNet b1.58 uses a 2-bit representation that
wastes one of the four possible values to store three states. TQ1.0 uses a
true Base-3 packing at 1.6 bits per weight. That is a 20 percent reduction
in memory over the naive 2-bit representation. For a 7B model at 1.58 bits
versus 1.6 bits, the difference is approximately 18 MB. Not decisive alone,
but it compounds with every other saving in the runtime.

## The C ABI

Every kernel is a C function with C linkage and a kiln_ prefix. The ABI is
stable. Adding a new kernel is allowed. Changing an existing signature is
not.

The packer:

  int kiln_tq1_0_pack(
      const float* src,   // input, length n, values in {-1.0, 0.0, +1.0}
      size_t n,           // number of elements
      uint8_t* dst        // output, length ceil(n/5)
  );

Returns 0 on success, non-zero on error. The only error condition is a
non-zero pointer requirement. Both src and dst must be non-null when n > 0.

The unpacker:

  int kiln_tq1_0_unpack(
      const uint8_t* src, // input, length ceil(n/5)
      size_t n,           // number of elements to produce
      float* dst          // output, length n, values in {-1.0, 0.0, +1.0}
  );

Returns 0 on success, non-zero on error.

Both functions are pure. They do not allocate. They do not touch global
state. They are thread-safe.

## The Rust FFI Contract

The Rust wrapper lives in crates/kiln-kernels. It exposes a safe API that
mirrors the C ABI.

  pub fn pack(src: &[f32]) -> Vec<u8>;
  pub fn unpack(src: &[u8], n: usize) -> Vec<f32>;

The safe API panics on invalid input. The invalid inputs are:

  - a value in src that is not in {-1.0, 0.0, +1.0}
  - an empty src slice
  - an unpack n that is larger than 5 times the byte length

The panic messages are explicit. They name the offending value and the
offending index. This is the Rule KILN-E4 principle applied to the kernel
boundary: when the API is misused, the error says exactly how.

The FFI layer uses std::os::raw::c_float and std::os::raw::c_int. These
are the Rust aliases for the C types. They are stable.

## The Build Process

The kernel layer uses build.rs to invoke the C compiler. No CMake. No
external build orchestration.

The build.rs file does:

  1. Locate cpp/kernels/*.c relative to CARGO_MANIFEST_DIR.
  2. Invoke the cc crate with the appropriate flags.
  3. Link the resulting static library into the crate.

The cc crate handles cross-platform flag differences. On Linux it uses
gcc or clang. On macOS it uses clang. On Windows it uses cl.exe or clang.
The cc crate is a build dependency of kiln-kernels only. It is not a
runtime dependency.

Why not CMake. CMake is a build system for C and C++ projects. KILN is a
Rust project with a C kernel. Adding CMake means two build systems, two
dependency graphs, and two places to look when a build fails. The cc crate
is the standard Rust way to integrate C. One command, cargo build, builds
everything. That is the correct trade-off for a project whose core
principle is one binary per platform.

## SIMD strategy

The first version of every kernel is scalar. It is correct, it is slow, it
is the reference. Every optimized version must produce bit-identical output
to the scalar version.

The optimized versions land in this order:

  1. AVX2. For Tier 0 through Tier 2 x86 machines.
  2. AVX-512. For Tier 5 and Tier 6 x86 machines.
  3. NEON. For Tier 3 Apple Silicon and Tier 4 ARM machines.
  4. WASM SIMD. For the browser fallback.

Each optimized path is gated on a runtime CPU feature check. The check
uses the is_x86_feature_detected macro in Rust. If the feature is not
present, the kernel falls back to scalar. This is what makes the one
binary per platform principle work: the same binary runs everywhere, and
picks the best path for the machine it lands on.

## Testing strategy

Every kernel has three tests:

  1. Round trip. Pack a known array, unpack it, verify equality.
  2. Boundary. Pack arrays of length 0, 1, 4, 5, 6, 100, 1000.
  3. Property. Pack a random array of length 10000, unpack it, verify
     equality. Run 100 iterations with different seeds.

The SIMD path is tested by forcing the feature flag and running the same
three tests. If the SIMD path and the scalar path produce different bytes,
the test fails. This is the Rule KILN-E13 principle: the anchor proves the
path landed, the differential test proves it is coherent.

## Format versioning

The TQ1.0 format is version 1.0. It is not expected to change. If it does,
the new version is TQ1.1 and the version byte is added to the model
manifest. The runtime refuses to load a model with an unsupported version.

The version byte does not exist in TQ1.0 because there is no ambiguity. A
TQ1.0 file is a byte string where every byte is in [0, 242]. If a future
format needs a version byte, it goes at the start of the tensor block in
the UMF container, not inside the packed byte stream.

## Open questions

  - How does the packer handle NaN and infinity in the input? The current
    design panics. A production version may want to clamp instead. The
    decision is deferred to the first real model pack.
  - How does the unpacker handle byte values in [243, 255] in the input?
    The current design treats them as an error. The decision is to keep
    this behavior. Any byte in that range indicates a corrupted file.
  - How does the kernel layer interact with the UMF container? The UMF
    stores packed bytes. The kernel layer packs and unpacks them. The
    boundary is the byte string. Documented here so it does not get
    confused later.

## Version history

2026-10-03 — Initial draft. TQ1.0 format, C ABI, Rust FFI contract, build
process, SIMD strategy, testing strategy, format versioning, three open
questions.
