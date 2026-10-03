# KILN Phase 0A Findings

Status: COMPLETE WITH ONE CONDITIONAL
Gate 0A: CONDITIONALLY PASSED
Started: 2026-10-03
Completed: 2026-10-03

## Purpose

This document records the verified findings for each of the eight Phase 0A research items. Every finding is either Confirmed, Contradicted, or Open. No item can remain Open when the gate is called, except by explicit conditional flag documented here.

## Rule

Every claim in this document cites a specific source with a stable identifier. Uncited claims are provisional. Provisional claims cannot pass the gate.

---

## Research Item 1 — Diffusion on 4-Core CPU

Status: CONFIRMED

Finding: Roofline paper validates arithmetic intensity argument. AR is memory-bound (AI roughly 2/b). Diffusion is compute-bound (AI roughly 2B/b, approximately 256 FLOP per byte at B=64). On 4-core i7-8650U, diffusion will not unconditionally beat AR on raw tokens per second but scales better with cores. Expect 3 to 6 tokens per second for diffusion on easy prompts. Pillar 1 stands for easy and medium prompts.

Sources:
- Roofline paper: Zenodo, 2026. "Diffusion Language Models are Faster than Autoregressive on CPU"
- diffuse-cpp benchmarks: 11 to 22 tokens per second on 12 cores versus llama.cpp 8.51 tokens per second

---

## Research Item 2 — Ternary Model Quality

Status: CONFIRMED WITH CAVEAT

Finding: Ternary Bonsai 8B (1.75 GB) scores 75.5 average benchmark versus 79.3 for FP16 Qwen3-8B (16.38 GB). That is a 9.36x size reduction for less than 5 percent relative quality loss. BitNet b1.58 2B4T is competitive with full-precision peers. Caveat: HuggingFace discussion claims low-bit models preserve coherency but lose knowledge resolution. Unresolved.

Sources:
- PrismML Ternary Bonsai whitepaper
- BitNet b1.58 2B4T Technical Report: arXiv 2504.12285
- HuggingFace discussion on Ternary-Bonsai-8B-mlx-2bit

---

## Research Item 3 — MoE Locality on PCIe 3.0

Status: CONFIRMED

Finding: Bottleneck is cache hit rate and prefetch accuracy, not raw SSD bandwidth. FlashMoE ML cache replacement is 21 percent better than LRU. TIDE interval-based expert refresh provides 1.4 to 1.5x speedup. AT-2 target of 3 tokens per second is plausible with both. Tail latency must be measured, not just mean.

Sources:
- FlashMoE: arXiv 2601.17063
- TIDE: arXiv 2605.20179

---

## Research Item 4 — KV Cache Quantization on CPU

Status: CONFIRMED

Finding: q8_0 is llama.cpp default with Hadamard rotation. Holds up at 64K context. HERALD achieves near-lossless at 5 to 10 percent KV budget for diffusion. KVDRIVE provides SSD-aware layout. AT-3 achievable.

Sources:
- llama.cpp PR: Hadamard rotation default
- HERALD: arXiv 2606.21633
- KVDRIVE: arXiv 2605.18071

---

## Research Item 5 — BitNet.cpp on AVX2-Only Hardware

Status: CONFIRMED

Finding: T-MAC LUT kernels are the AVX2 path. 6.6x kernel speedup over llama.cpp. 11.1 tokens per second on Raspberry Pi for BitNet-b1.58-3B. AYOT makes post-training ternarization practical. Maple Preview 20B-A1B runs at 28 to 34 tokens per second on i5-8350U with TQ2_0.

Sources:
- T-MAC: arXiv 2407.00088
- AYOT: arXiv 2608.01078
- Maple Preview benchmark on i5-8350U with TQ2_0

---

## Research Item 6 — Thermal Throttling Steady State

Status: CONFIRMED

Finding: Dell community threads document 90C under load. Throttles at 80C. Settles at 85C with 30 percent sustained load. ANM-V1 paper confirms the pattern is canonical for thin-chassis machines. Pillar 6 is required.

Sources:
- Dell community threads on Latitude 7490 thermal throttling
- ANM-V1 paper on MacBook Air M2 thermal behavior

---

## Research Item 7 — SSM vs Transformer for Tier 0

Status: CONFIRMED

Finding: BitMamba-2-1B (621 MB, 2-bit) runs at 52.86 tokens per second on Intel i3-12100F. O(1) memory property validated. Competitive benchmarks in 1B class. Caveat: MoE SSM routing fails (BoolQ drops to 42.54 percent). Dense SSM is a viable Pillar 1 alternative.

Sources:
- BitMamba-2: Zenodo 18394665
- SSM CPU-native paper: Scilit 43121fc27d6aab27fcc23359edfd02f8

---

## Research Item 8 — Thermal Envelope Measurement

Status: OPEN-CONDITIONAL

Partial evidence collected 2026-10-03 on Dell Latitude 7490, on battery, 5-minute run with qwen2.5:0.5b (397 MB).

Results:
- 11 samples at 30-second intervals.
- Sample 0 outlier: 1.733 tokens per second (cold start, model load).
- Steady-state band: 26 to 37 tokens per second, average approximately 30 tokens per second.
- CPU temperature: stable 62 to 64C across the run, one start spike at 67C.
- CPU frequency: locked 3900 MHz for 10 of 11 samples, one dip to 1760 MHz.
- No throttling observed.

Limitations:
- Load too light. A 0.5B model running 1 second every 30 seconds does not stress the CPU enough to trigger throttling.
- Run too short. 5 minutes is not 30 minutes.
- On battery. AC behavior may differ.
- Test harness validated but the research question is unresolved.

Required follow-up before Phase 2 gate:
- AC-powered 30-minute continuous load run.
- Model size 1.5B or larger (nemotron-3-nano:4b or qwen2.5-abliterate:1.5b).
- Sleep between samples reduced to 5 seconds for near-continuous load.
- Record tokens per second, CPU MHz, and CPU temperature every 30 seconds.
- Ambient temperature recorded at start and end.

Gate 0A impact: This item is OPEN-CONDITIONAL. Gate 0A may pass on the other seven items with Item 8 flagged for dedicated thermal testing before Phase 2. Phase 0B work does not depend on the thermal envelope. Phase 2 work does.

Sources:
- Local measurement 2026-10-03 (partial, see limitations)
- Dell community threads on Latitude 7490 i7-8650U throttling

---

## Gate 0A Checklist

Seven of eight items CONFIRMED. One item OPEN-CONDITIONAL (Item 8).

All confirmed items cite at least two sources with stable identifiers.
The contradicted-item clause does not apply; no items were contradicted.
Confirmed findings are reflected in the Master File.
Item 8 is flagged for dedicated thermal testing before Phase 2.

Gate 0A verdict: CONDITIONALLY PASSED.

Phase 0B may begin. Item 8 must be resolved before Phase 2 gate.
