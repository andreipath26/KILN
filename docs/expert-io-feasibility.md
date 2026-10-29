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

## Phase 5B real-file cache replay

After building the CLI, compare no-cache, LRU, LFU and adaptive cache policies against the same deterministic synthetic trace and identical retained-byte budgets. To replay a model-derived trace against a validated variable-size index, supply `--index-file /path/to/experts.index.json` and a layer-aware `--trace-file` captured from a router or `CachedExpertBlobReader`:

```sh
cargo run -p kiln-bench -- expert-cache \
  --file /path/to/representative-large-file.bin \
  --block-bytes 1048576 \
  --requests 10000 \
  --cache-mib 256 \
  --workload phase-shift \
  --seed 42 \
  --json
```

Workloads are `phase-shift`, `stationary-hot`, `uniform`, and `round-robin`. The command partitions the file into fixed-size synthetic expert blocks, replays the same trace for each policy, and reports elapsed time, request rate, hit rate, estimated backing bytes read, cache bytes and evictions. It performs real file reads on the machine where the command runs, but the access trace is synthetic and the OS page cache is uncontrolled. Run it on the target Dell for target-machine data; do not interpret a CI-host result as Dell performance.

Without `--index-file`, the benchmark partitions the file into fixed-size synthetic layer-0 blocks; `--trace-file` can then contain flat IDs such as `[0, 0, 3, 7, 3]`. With `--index-file`, the file must be a JSON-serialized `ExpertTable` whose validated entries map each `(layer, expert_id)` to one contiguous byte range in the blob. The trace can then contain flat IDs, `[layer, expert_id]` pairs, or objects such as `{"layer": 1, "expert_id": 7}`; every key must exist in the index. This is not a GGUF index extractor: the blob and index must already be prepared in KILN's current single-range-per-expert layout.

When an index is supplied, estimated backing bytes are the sum of actual indexed range lengths for cache misses; without an index, the fixed block size is used. These values represent bytes requested from the file API, not physical-device counters. The cache budget counts retained expert bytes only; returned buffers are outside that budget. Trace-file length determines the request count and `--workload` is ignored. The benchmark replays the trace but does not itself collect live llama.cpp routing events.

## Phase 5C predictor primitive (planning only)

The `kiln-mem::ExpertPrefetcher` learns one-step transitions from an observed `ExpertKey` sequence and ranks candidates by empirical conditional frequency. The caller supplies candidate block sizes. It enforces a cap on total tracked speculative requests (in-flight plus ready metadata) and a separate in-flight byte budget and exposes counters for scheduled predictions, cap rejections, useful, late, unused and cancelled requests. The integration caller must call `mark_ready` only after a read completes before demand; a demand that arrives while still in-flight is counted as late, not useful. Ready expert bytes are accounted for by the cache layer, not by the in-flight byte budget; the planner bounds ready metadata as well. Tests cover ranking/confidence, budget enforcement, outcome accounting and stale-plan cancellation.

This is deliberately a predictor/planner primitive, not asynchronous I/O: it does not submit reads, fetch expert bytes, evict cache entries, observe live llama.cpp router events, or prove overlap. The integration layer must call `observe` for actual expert accesses, use the model's validated expert index for sizes and identities, execute returned plans, and report outcomes accurately. Compare against no-prefetch and naive baselines on the same captured trace before enabling it in inference.

### Replaying prefetch predictions against an access trace

```sh
cargo run -p kiln-bench -- prefetch-replay \
  --trace-file /path/to/expert-ids.json \
  --block-bytes 1048576 \
  --top-k 2 \
  --max-pending 8 \
  --max-inflight-mib 16 \
  --latency-requests 1 \
  --json
```

The input can be a JSON array of flat IDs (e.g. `[0, 1, 0]`, interpreted as layer 0), `[layer, expert_id]` pairs (e.g. `[[0, 1], [1, 3], [0, 1]]`), or `{layer, expert_id}` objects. Each candidate is assigned the fixed size given by `--block-bytes`; the candidate universe is limited to experts appearing in the trace. The replay reports top-k next-event coverage/precision and planner outcomes. `--latency-requests` simulates read completion in units of trace events; it is not a wall-clock I/O measurement. This is a diagnostic predictor experiment only. It does not issue file reads, model live routing, account for cache evictions of ready prefetched blocks, or measure tokens/s.

## Bounded access trace capture from the cache-backed reader

Call `CachedExpertBlobReader::enable_access_trace(max_events)` before a workload to record successful expert reads, including cache hits. Recording is disabled by default. The trace is bounded: once the configured event limit is reached, additional accesses increment `dropped_events` instead of growing memory. Use `access_trace_snapshot()` to inspect the recorded events and dropped count, or `write_access_trace_json(path)` to export a JSON array accepted by `kiln-bench prefetch-replay --trace-file`. The expert-cache command currently accepts flat block IDs only; mapping `(layer, expert_id)` into file block IDs requires the actual model index and must not be guessed.

This only records accesses that pass through `CachedExpertBlobReader`. The current llama.cpp backend does not route its internal MoE expert reads through this reader, so this API does not yet capture live llama.cpp routing. It creates a concrete trace-capture seam for a KILN-native expert-loading path and for tests.

The predictor's transition table is also hard-bounded (65,536 source keys and 65,536 distinct transitions). When the edge table is full, it stops adding new transitions and continues counting observed transitions for already-tracked sources, keeping probabilities' denominator honest without unbounded map growth. Duplicate candidates and keys already in flight or marked ready are not scheduled twice.
