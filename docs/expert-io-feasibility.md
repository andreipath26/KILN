# Expert-block I/O feasibility probe

## Purpose

Phase 5A asks whether the target machine's storage path can feed representative MoE expert blocks at a useful rate. This probe establishes a repeatable, portable seek/read baseline before introducing asynchronous I/O, cache policies, prefetching, or expert routing.

## Command

```sh
cargo run -p kiln-bench -- expert-io \
  --file /path/to/experts.bin \
  --block-bytes 1048576 \
  --requests 256 \
  --runs 5 \
  --seed 42 \
  --json
```

The file must contain at least one complete block. Sequential requests walk blocks in file order. Random requests use a deterministic generator seeded by `--seed`. Both patterns use the same block size, request count, and repetition count.

## Output

The JSON report includes file path and size, OS and architecture, block size, request count, repetitions, seed, total bytes read, throughput, p50/p95 request latency, and a checksum to keep the reads observable.

## Interpretation limits

- The implementation is a correctness-first portable `seek + read_exact` baseline, not `io_uring`, `O_DIRECT`, or overlapped I/O.
- OS page cache state is not controlled; later repetitions can be warm. Do not label this a cold-storage benchmark.
- Random offsets are block-aligned and selected from complete blocks; this is a synthetic access pattern, not a real router trace.
- The probe does not parse or validate `experts.index`, measure concurrent I/O/compute overlap, measure peak memory, or run a model.
- A successful result is not the Phase 5A acceptance gate. Compare against a pinned representative expert layout on the Dell, then measure end-to-end inference before deciding whether an I/O backend or cache/prefetch policy is worthwhile.

## Phase 5B cache foundation

The current branch also adds a byte-bounded, thread-safe expert cache in `kiln-mem` and a `CachedExpertBlobReader` in `kiln-io`. The cache supports LRU, LFU with recency tie-breaking, and adaptive frequency aging. It exposes retained-byte, hit/miss, entry, and eviction counters. Oversized experts are returned from the backing blob but are not retained; invalidation and clearing release cache membership bytes.

The reader rechecks a key after acquiring its serialized file-reader lock, preventing concurrent cold requests for the same cacheable key from all issuing duplicate reads. Cache policy tests cover byte accounting, eviction, oversized entries, duplicate-key insertion, concurrent access, and invalidation. Synthetic trace tests cover phase shift, stationary hot-set, uniform and round-robin regimes.

Important limits:

- This is an initial cache primitive and reader wrapper; no live KILN model execution path is wired to it yet.
- The byte budget bounds bytes retained by the cache, not caller-owned copies returned by `read_expert`; consumers must account for those buffers separately.
- The file reader serializes cold reads. Async I/O, routing-informed prefetch, and compute overlap remain later phases.
- Synthetic traces are not model-routing traces. Do not choose a production policy or claim a speedup until measured traces and target-hardware results exist.
