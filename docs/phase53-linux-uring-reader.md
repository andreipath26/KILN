# Phase 5.3 — Linux io_uring expert reads and SQPOLL fallback

A Linux-only `LinuxUringExpertReader` now opens an indexed expert blob, validates the complete `ExpertTable` before reading data, and issues `IORING_OP_READ` requests through the upstream `io-uring` Rust crate. It reads bounded 64 KiB segments and handles short reads, oversized host offsets, missing experts and buffer-length mismatch.

`LinuxUringExpertReader::open(path, table, request_sqpoll)` supports three observable modes:

- `IoUringSqPoll`: opt-in SQPOLL setup succeeds. CPU/kernel permissions and SQPOLL scheduling may make this slower on small machines.
- `IoUring`: regular io_uring setup succeeds, including fallback when optional SQPOLL setup fails.
- `BufferedFallback`: ring creation or a ring read fails; use safe buffered positioned reads. An unavailable or denied io_uring subsystem must never stop KILN from reading experts.

This is a **native read-path implementation**, not yet the entire Phase 5.3 scheduler. The existing bounded background worker and exact-router Commit Guard remain available as a portable overlap and correctness path. The next integration must route their source reads through this backend, batch independent expert I/O when beneficial, and compare SQPOLL against plain buffered reads under the Dell's thermal and I/O constraints.

## Safety

io_uring retains raw destination pointers until completion. Each request uses a distinct private 64 KiB kernel buffer, never a borrowed user slice. The buffer is freed after its CQE is reaped. If submission fails ambiguously after an SQE is pushed, the code disables io_uring, deliberately retains that one 64 KiB kernel buffer rather than freeing memory that the kernel might still access, and falls back to positioned reads. This failure-path sacrifice is bounded to one buffer per reader because the ring is disabled after the first ambiguous failure. It must be covered by future fault-injection tests.

No direct I/O (`O_DIRECT`) is enabled by default. Sequential SSD bandwidth, page-cache hits, random-read pressure and total RAM consumption are distinct metrics. The Linux read gate does **not** establish throughput, SSD bytes/token, SQPOLL superiority or the Tier 0 15 GB peak-memory acceptance target.

## Validation

The Linux unit test reads two distinct indexed expert ranges, including a range larger than a 64 KiB request, compares exact source bytes, verifies wrong buffer/key rejection, and permits an ordinary buffered fallback if CI kernels prohibit io_uring. A second test checks bad range rejection before ring setup.

Execute on Dell with a pinned real model/expert index and compare regular ring, optional SQPOLL and the existing buffered reader. Integrate with actual llama.cpp MoE execution only after its routing and fallback tests pass.

## Native io_uring background prefetch integration

The same phase now includes a Linux-only `AsyncGuardedExpertCoordinator::new_linux_uring(...)` factory. It opens the native ring reader and authoritative cache reader from the **same expert path and validated ExpertTable**. The background worker can use the real `IORING_OP_READ` kernel path, falling back to buffered positioned reads when ring setup or operation is unavailable. The epoch-scoped Commit Guard accepts only completed proposed blocks for the matching token, then exact router keys authorize output. Missing experts are always loaded from the authoritative source.

An integration test creates actual expert file bytes, stages two predictions, deliberately routes to one correct and one missing expert, checks exact byte equality and router order, and verifies guard bytes are released. The optional SQPOLL setup can be denied by CI; correctness still passes through fallback.

The live llama.cpp router and model compute graph are still **not** connected to this coordinator, so no end-to-end streamed inference or Tier 0 speed claim is made.
