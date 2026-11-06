# Phase 3.5 — process memory evidence

The `kiln bench <model.gguf> --json` command reports actual Linux process memory snapshots without an external monitor.

## Measurements

- `process_memory.before_load`: current `VmRSS` and process-to-date `VmHWM` from `/proc/self/status` before loading the model.
- `process_memory.after_load`: same fields immediately after model loading.
- `process_memory.after_benchmark`: same fields after all warm-up and measured generations.
- Each entry in `runs` gains `process_rss_kib_after_run` and `process_peak_rss_kib_to_date`.

Memory values use **KiB**, while the peak counter is a *monotonic high-water mark* since process start. It is **not** a per-run peak. The values are `null` on unsupported platforms or when Linux procfs is unavailable. Parsing tests cover proper units, exact field keys, and missing records.

### What these metrics do **not** prove

RSS does not include every source of memory pressure, including OS filesystem cache held outside process RSS. It also does not quantify temporary I/O buffers in another process or guarantee that independent components enforce a shared RAM budget. The numbers are useful for model benchmarking and regression detection, but **not sufficient** for the Phase 5 acceptance requirement of safely operating below 15 GB of peak total memory on Tier 0. Add machine-level MemAvailable, page cache pressure, swap activity, and workload-integrated buffer accounting to the final Dell acceptance procedure.

These metrics add observability only; they do not claim any new inference speed or model capability. Capture and compare model file hashes, KILN and llama.cpp commits, threads, batch, context length, warmup, thermal state and repetition counts as required by the master roadmap.
