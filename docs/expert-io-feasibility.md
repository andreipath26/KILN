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
