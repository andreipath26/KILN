KILN Master File — Complete Session State
Version 4.0 — 2026-10-03
Purpose of This File

This file exists so that if the working session is interrupted, the entire project state can be restored by pasting it into a fresh conversation. It contains the mission, the principles, the architecture, the reasoning behind every decision, the rejected alternatives, the external review findings, the pressure-test results, the open research questions, the roadmap, the rulebook, and instructions for resuming.

This file is not a summary. It is the complete session state, including every detail from today's session.
How To Use This File

Save it as KILN-master.md. When resuming, open a new chat and paste the entire file, then add the line: "Read this file. You are the co-architect of KILN. Confirm you are up to speed, then wait for my next instruction."
Table of Contents

Section 1 — Mission and Origin
Section 2 — Prime Directives
Section 3 — Hardware Tiers
Section 4 — The Six Pillars
Section 5 — Six-Layer Runtime Stack
Section 6 — The Design Journey
Section 7 — Rejected Alternatives
Section 8 — External Review Findings
Section 9 — Pressure-Test Results
Section 10 — Corrected Numbers and Constraints
Section 11 — Phase 0A Research Items
Section 12 — Implementation Roadmap
Section 13 — Acceptance Tests
Section 14 — Rulebook
Section 15 — Testbed Migration Ladder
Section 16 — Source Project Reference List
Section 17 — Change Log
Section 18 — File Format Rules
Section 19 — Current Status
Section 20 — Open Questions
Section 21 — Files Saved So Far
Section 22 — Instructions for the Resuming AI
Section 23 — Project Lead Profile
Section 24 — Research Collaboration Notes
Section 25 — Knowledge Cutoff Disclosure
Section 26 — Conversation History of Today's Session
Section 27 — End of Master File
Section 1 — Mission and Origin

Codename: KILN.

Mission: Build the fastest local LLM runtime on Earth. One binary per platform, one model file, one API. Runs flawlessly on a 2018 Dell laptop, an 8xH100 server, and a ThinkPad in Sydney. No cloud. No GPU required. No compromise on accuracy.

Origin: The project began with a Dell Latitude 7490 (i7-8650U, 16GB DDR4-2400, no GPU, NVMe SSD). The user wanted to run local LLMs larger than 9B parameters with fast responses and accurate output. Standard tooling does not achieve this on the target hardware. The user rejected the framing that this hardware is "not enough" and asked for an architecture that makes it sufficient.

The user's principle throughout: AI must be accessible to the masses, including people on hardware five to ten years old. No one should be excluded because they cannot afford an upgrade.

Naming rationale: The codename KILN was chosen for a kiln — a furnace where raw materials are transformed under controlled heat and pressure into something stronger. The project transforms raw techniques from research papers and open-source projects into a unified runtime.
Section 2 — Prime Directives

    Lowest-spec machine is the acceptance test. If it runs on a Dell Latitude 7490 (i7-8650U, 16GB DDR4-2400, no GPU, NVMe SSD), it runs everywhere.

    One binary per platform, one model file, one API. No per-machine forks. No user-side compilation.

    Per-layer dispatch, not per-machine modes.

    Accuracy preserved. Every optimization is lossless or within documented tolerance on standard benchmarks.

    Local by default. No network calls unless the user opts in.

    Open-core. Core runtime is MIT or Apache-2.0.

    Fail-safe. Any backend failure falls back to CPU. Any model load failure reports the reason and exits cleanly.

    The masses come first. Every decision is tested against the lowest-spec machine.

    No architecture change to accommodate one machine. If a flag is required for the runtime to work on a specific machine, that is a Capability Registry bug. Fix the registry, not the architecture.

    Optional flags and toggles are allowed. Per-machine builds are forbidden. The test: if the runtime requires a flag to work on a specific machine, the architecture has failed.

Section 3 — Hardware Tiers

Tier 0. Dell Latitude 7490. i7-8650U, AVX2 only, 16GB DDR4-2400, no GPU, NVMe SSD. Acceptance test.

Tier 1. ThinkPad T14. Ryzen 7, 32GB DDR4, Intel iGPU.

Tier 2. Gaming laptop. i9 plus RTX 4070, 32GB DDR5, 8GB VRAM.

Tier 3. MacBook Pro M3. 18GB unified memory, ANE plus GPU.

Tier 4. Snapdragon X Elite. 16GB, Hexagon NPU.

Tier 5. Xeon workstation. 128GB, 2x A6000.

Tier 6. EPYC server. 512GB, 8x H100.

Every tier now has a performance subclass: A (sustained), B (throttling), C (severely throttling). Tier 0 is subclass B or C by default. The runtime determines the subclass at first run and updates it dynamically.
Section 4 — The Six Pillars

Pillar 1 — Attention-Guided Diffusion Generation. Fork diffuse-cpp. Add CLAD-style cluster-level parallel decoding. Add inter-step KV cache reuse. Add attention-guided denoising exploiting sparse, temporally consistent attention patterns. Add Eso-LMs for exact KV caching. Add TIDE for MoE-dLLM interval-based expert refresh. Add HERALD for block-dLLM KV offloading. Add ADLM for anchored diffusion quality. Add BitMamba-2 as SSM alternative.

Pillar 2 — Ternary and Mixed-Precision Quantization. ATLAS TQ1.0 packing. int4 FFN mode. T-SAR-style in-register LUT generation. Native BitNet b1.58 support. Add T-MAC for AVX2 LUT kernels. Add MAGNET and Genkidama for ternary validation. Add Trident as pure-Rust ternary training. Add Camada as native ternary MoE. Add SmallThinker as target model family. Add AYOT for differentiable ternarization in the PTQ regime.

Pillar 3 — Predictive MoE Streaming with EDGE0-Style Prerouter. Per-layer prerouters trained to predict next-layer routing one token ahead. Expert-contiguous storage layout. Block-level prefetching with bounded in-flight blocks. Add FlashMoE for ML-based cache replacement. Add FIRM-MoE for fine-grained expert decomposition. Add ExpertFuse for model-level expert merging. Add TIDE for interval-based expert refresh. Add storage.llm as the closest existing implementation.

Pillar 4 — SSD-Backed KV Cache with q8_0 Default. llama.cpp upstream q8_0 KV cache as default. q4_0 for aggressive cases. HiFC pSLC only for enterprise tiers with write endurance headroom. Add HERALD for diffusion KV offloading. Add KVDRIVE for SSD-aware layout. Add RIS-Kernel for long-context sparse attention. Add RaBitQCache as rotated binary quantization candidate.

Pillar 5 — Constraint-Aware Registry and DAG-Based Scheduler. Capability Registry with conversion cost matrices, tensor residency state, transfer latency, synchronization overhead, and thermal state. Execution plan as a DAG. Add SlyOS as design reference. Add Ghidorah's ARCA approach for architecture-aware profiling. Add thermal history field for predictive throttling management.

Pillar 6 — Adaptive Performance Management. The runtime continuously measures CPU frequency, thermal state, RAM pressure, and SSD throughput. It adjusts the DAG execution plan in response. It never assumes a stable machine. The degradation ladder is: 100-80% baseline no change; 80-60% disable speculative decoding; 60-40% drop KV cache precision; 40-25% switch to smallest model; below 25% pause and notify. It preemptively switches if thermal history predicts throttling. It restores the original plan within 60 seconds of sustained recovery.
Section 5 — Six-Layer Runtime Stack

Layer 6. User API. CLI, Ollama-compatible REST server, Python bindings in embedded and client mode.

Layer 5. Automatic Mode Selector. Inspects model architecture and hardware profile, produces a per-layer DAG execution plan.

Layer 4. Execution Plan. Nodes are computations annotated with precision, placement, and algorithm. Edges are data transfers with measured cost.

Layer 3. Constraint-Aware Registry. Every backend declares supported operations, supported precisions, memory model, available memory, device bandwidth, conversion costs, residency state, transfer latency, synchronization overhead, and thermal state.

Layer 2.5. Performance Monitor. Between the Memory Hierarchy Manager and the Constraint-Aware Registry. Exposes live performance envelope to the scheduler. The scheduler reads it before every significant decision.

Layer 2. Memory Hierarchy Manager. Manages four storage tiers as one pool. CPU L3 cache, RAM, NVMe SSD, optional VRAM.

Layer 1. Backends. CPU with AVX2 and AVX-512, CUDA, ROCm, Metal, Vulkan, WebGPU, OpenVINO for Intel NPU, CoreML for Apple ANE, QNN for Qualcomm Hexagon, WASM.
Section 6 — The Design Journey
6.1 The Initial Problem

The user asked how to run local LLMs larger than 9B on a Dell 7490. The first response identified the constraints: CPU-only inference is memory-bandwidth-bound. A dense 30B model must read 30B weights per token.

The first round of research identified Mixture-of-Experts (MoE) as the key architecture. MoE models have many parameters but only activate a small subset per token.
6.2 The Shift to SSD Streaming

The second round identified that even 3B active parameters may not fit in 16GB RAM alongside the OS and applications. The solution: stream expert weights from NVMe SSD on demand. Projects identified: SSD-LLaMA, Micro-Expert-Router, EDGE0, llama.cpp PR 25294, Guanaco, runNburn, Hummingbird.

Key insight: the bottleneck is not raw SSD bandwidth but expert cache hit rate and prefetch accuracy.
6.3 The Diffusion Insight

The third round identified that on CPU, autoregressive decode is memory-bound, while diffusion LM is compute-bound. This shifts the workload to a regime that scales across CPU cores. Project identified: diffuse-cpp. The roofline paper later confirmed this with formal arithmetic intensity analysis: AI_AR ≈ 2/b for AR and AI_diff ≈ 2B/b for diffusion, a 64x difference at B=64.

Key insight: the diffusion advantage depends on the number of denoising steps. Inter-step KV cache reuse provides 1.6x additional speedup.
6.4 The Ternary Insight

The fourth round identified that ternary quantization (1.58-bit) reduces weight memory by 10x, which reduces memory bandwidth pressure by 10x. Projects identified: sasori, FairyFuse, Sherry, ATLAS, BitNet.cpp. T-MAC later confirmed as the AVX2 LUT kernel library.
6.5 The KV Cache Insight

The fifth round identified that the KV cache competes with model weights for the 16GB RAM. Projects identified: HiFC, Dual-Blade, Ada-KV, HqeKV, SpeCache. Later additions: HERALD for diffusion KV offloading, KVDRIVE for SSD-aware layout, RIS-Kernel for long-context sparse attention.

Key insight from llama.cpp upstream: Hadamard rotation is applied to KV activations by default since April 2026, making q8_0 KV cache the standard.
6.6 The Predictive Prefetching Insight

The sixth round identified that expert streaming fails without accurate prediction. EDGE0 demonstrates 20.4 tok/s on 35B MoE on Mac mini. Later additions: FlashMoE for ML-based cache replacement, FIRM-MoE for fine-grained expert decomposition, ExpertFuse for model-level expert merging, TIDE for interval-based expert refresh, storage.llm as the closest existing implementation.
6.7 The Hardware Abstraction Insight

The seventh round identified that for the runtime to work on all tiers, it needs a hardware abstraction layer. Projects identified: MLC LLM, ZML/LLMD, LiteRT-LM, SiliconScavenger, npurun, OpenVINO GenAI, Ryzen AI, CoreML-LLM, CoreAI. Later additions: SlyOS device intelligence layer, Ghidorah ARCA approach.
6.8 The WHAXON Integration Discussion

The user has a separate project at github.com/andreipath26/WHAXON. WHAXON is a security orchestration tool that uses AI but is too slow for its purpose. Decision: KILN exposes an Ollama-compatible API. WHAXON adapts to consume it. The user was explicit: "We can modify whaxon to fit with this project if needed be, not the other way around."
6.9 The Neighborhood and Internet Mesh Discussion

The user asked about pooling resources across neighbors and the broader internet. Projects discussed: HearthNet, MeshMind, NVIDIA PAIR, Mesh LLM, Parallax, PlanetServe, EIGR-Infer, Blindference Node, mycellm, PolyLink TIQE. Decision: drop the internet mesh because latency kills the interactive experience.
6.10 The GPU Discussion

The user asked if KILN would work on GPUs. Confirmed yes. GPUs are first-class citizens. Caveat: On GPUs, KILN competes with well-optimized existing runtimes. The GPU advantage is narrower: models that do not fit in VRAM, long contexts, and heterogeneous systems. The "Challenging GPU Dominance" paper confirms CPU can outperform GPU for single-batch inference on small models.
6.11 The Multi-Machine Testbed Ladder

The user confirmed: use the Dell 7490 as a testbed until it is no longer viable, then migrate to other machines in a ladder.
6.12 The Master File Creation

The user asked for a master file that could restore the entire session state if the computer crashed. This file is the result. It was created in multiple iterations because earlier versions were truncated or lost detail.
6.13 The Deep Research Round

The user asked for a deep search especially targeting scientific research papers to avoid missing anything before building begins. The search surfaced 18 papers and additional projects, all integrated into this version.

Key new findings: TIDE, HERALD, KVDRIVE, FlashMoE, SmallThinker, MAGNET/Genkidama, "Challenging GPU Dominance", T-MAC, SlyOS, Ghidorah, ExpertFuse, storage.llm, Trident, BitMamba-2, Camada, AYOT.
6.14 The Second Deep Research Round

The user asked for one more broad search targeting GitHub projects and papers from the last three months. The search surfaced Ghidorah, ExpertFuse, storage.llm, Trident, BitMamba-2, Camada, and additional findings.

The most important new finding: BitMamba-2 and Camada are not transformer models. If SSM at 1B parameters can match 7B transformer quality, the Tier 0 strategy changes fundamentally.
6.15 The Thermal Throttling Discovery

The user asked about real-time responsiveness for coding. The research surfaced specific Dell 7490 thermal data: the i7-8650U reaches 90°C under sustained load, thermal throttling enables at 80°C, and the CPU settles at 85°C with only 30% sustained load after throttling. Multiple Dell community threads document this as a known issue with the model, not an edge case.

Key insight: Every Tier 0 machine throttles. The runtime must expect it. This led to the creation of Pillar 6 (Adaptive Performance Management).
6.16 The Model Recommendation Discussion

The user asked what model would be best for coding with a near real-time feel. The recommendation was Parable-Qwen3-4B (Q4_K_M), a specialized coding agent fine-tuned on real Claude agent traces. Its 2.5 GB quantized size leaves 12+ GB of RAM free for context and OS.

The rationale: For the Dell 7490, prioritizing low active parameters is the only way to achieve the speed needed for interactive coding. Larger "coder" models like Qwen3-Coder-30B-A3B run at only 5.1 tok/s on an 8-core Ryzen 7 PRO 8700GE when limited to a RAM footprint that would fit the Dell.
6.17 The Final Deep Research Round

The user asked for one final in-depth search before amending the roadmap. The search confirmed FlashMoE, Fast-dLLM, Edge0, and BitNet.cpp as validated techniques. It surfaced AYOT as the method that solves the post-training ternarization problem. It also surfaced the Maple Preview 20B-A1B benchmark: 28-34 tok/s on an i5-8350U (same generation as the i7-8650U, 4 cores, AVX2) with TQ2_0 ternary quantization and a 5.5 GB file.

This is the strongest evidence yet that the Tier 0 targets are conservative. The i5-8350U is a 2019-era laptop CPU with a chassis similar to the Dell 7490.
6.18 The Adaptive Performance Pillar Addition

The user confirmed that the thermal issues on the Dell 7490 must be designed into the architecture so that no user has problems with the project if they run into the same issues. This led to the addition of Pillar 6 and the specific degradation ladder.
Section 7 — Rejected Alternatives

Internet mesh. Rejected because internet latency destroys the interactive experience.

Cloud offload. Rejected because it violates the no-cloud principle.

Per-machine builds. Rejected because it violates the one-binary principle.

Hardware-specific forks. Rejected for the same reason.

User-side compilation. Rejected because it excludes non-technical users.

Abandoning Tier 0 targets. Rejected by the user.

Accepting defeat. Rejected explicitly by the user.

Uniform 2-bit MoE quantization. Rejected because it causes significant quality degradation.

Continuous SSD KV cache writes on laptop drives. Rejected because of write endurance.

Speculative decoding by default on Tier 0. Rejected because acceptance rate is often below the threshold.

Per-layer dispatch of diffusion vs AR. Clarified as not a real proposal.
Section 8 — External Review Findings (Full)
8.1 Gemini Review

Source: https://share.gemini.google/BVHdLcd4BaI8

Key findings kept:

The i7-8650U has no AVX-512. Only AVX2. The FairyFuse AVX-512 kernel path cannot run on Tier 0.

The Dell 7490 NVMe throughput is approximately 2 GB/s, not 3.2 GB/s.

The Dell 7490 thermally throttles under sustained load.

Speculative decoding is not guaranteed to help on CPU. Below ~37% acceptance, even γ=1 is counterproductive.

Key findings discarded:

Gemini assumed AVX-512 was available on Tier 0.

Gemini assumed 3.2 GB/s NVMe throughput.

Gemini treated speculative decoding as a guaranteed win.
8.2 ChatGPT Review One

Key findings kept:

The spec mixes three different kinds of optimization as though they were interchangeable.

The Capability Registry must describe constraints, not just capabilities.

AT-3 as stated is not generally feasible.

"No compromise on accuracy" plus arbitrary extreme quantization is contradictory.

SSD KV cache as a continuous write target is dangerous on a laptop drive.

BitNet.cpp, Fast-dLLM, SPEED, MoE-Infinity, and the recent llama.cpp MoE streaming PR are missing from source references.

Key findings discarded:

"Per-layer dispatch of diffusion vs AR is fundamentally wrong." The spec never proposed this.

"One binary with no platform-specific backend code is not feasible." The spec already implies this.

"The combinatorial explosion is the single biggest risk." The biggest risk is the thermal plus I/O plus no-AVX-512 combination on Tier 0.
8.3 ChatGPT Review Two

Key findings kept:

The spec is not yet technically coherent as an end-to-end system for Tier 0.

Activation quantization is missing.

The SSD-as-RAM latency gap is real.

The MoE SSD streaming pillar is under-specified without locality invariants.

pSLC KV cache write behavior is likely to break AT-8 and endurance.

Key findings discarded:

"AT-2 is fundamentally impossible." Too absolute. EDGE0 demonstrates otherwise.

"Diffusion-first will struggle to hit 2-3 tok/s without ternary." Too pessimistic.

"Speculative decoding should be disabled by default on Tier 0." Threshold should be 0.40, not 0.60.
8.4 ChatGPT Review Three

Key findings kept:

Activation quantization is the missing pillar.

The SSD-as-RAM latency gap is real and the spec underestimates it.

The AT-2 bandwidth math: at 2 GB/s, need 4.1 GB/s for 5 tok/s.

The pSLC write endurance math: 7.08-day lifespan unconstrained.

Speculative decoding: below α=0.37-0.40 acceptance, even γ=1 is counterproductive.

Key findings discarded:

"AT-2 is fundamentally impossible." Same as Review Two.

"Diffusion-first will struggle to hit 2-3 tok/s." Same as Review Two.
8.5 Findings All Three Reviews Missed

The expert cache hit rate is the single number that determines AT-2 feasibility.

The inter-step KV cache is the diffusion equivalent of speculative decoding.

The write-back ring buffer is the correct pSLC mitigation, not a throttle.

The expert-contiguous storage layout is a Phase 0 format decision.

The bottleneck has shifted from SSD bandwidth to cache hit rate and prefetch accuracy.
8.6 Unreviewed External Source

A Claude share link was provided but was geo-blocked. This remains an open item.
Section 9 — Pressure-Test Results

Claim: i7-8650U has no AVX-512. Status: Confirmed.

Claim: Dell 7490 NVMe is ~2 GB/s. Status: Confirmed.

Claim: Dell 7490 throttles under sustained load. Status: Confirmed by multiple Dell community threads.

Claim: Speculative decoding below 37% acceptance is counterproductive. Status: Confirmed.

Claim: AT-3 requires 32 GB FP16 KV cache for 7B at 64K. Status: Confirmed.

Claim: q8_0 KV cache is now the llama.cpp default. Status: Confirmed.

Claim: EDGE0 achieves 20.4 tok/s on 35B MoE on Mac mini. Status: Confirmed.

Claim: MER achieves 71.9% hit rate at 25% residency, 0.551 tok/s. Status: Confirmed.

Claim: Diffuse-cpp achieves 27.7 tok/s on EPYC 12-core. Status: Confirmed for translation task.

Claim: Dream-7B inter-step cache provides 1.6x speedup. Status: Confirmed.

Claim: pSLC unconstrained KV caching has 7.08-day lifespan. Status: Confirmed.

Claim: Write-back ring buffer reduces WAF by >25x. Status: Confirmed.

Claim: Qwen3-30B-A3B activates 3.3B parameters per token. Status: Confirmed.

Claim: BitNet.cpp achieves 2.37-6.17x speedup on x86. Status: Confirmed for AVX-512 hardware.

Claim: Attention patterns in diffusion LMs are sparse and temporally consistent. Status: Confirmed.

Claim: 4-core diffusion scaling is roughly 4/12 of 12-core. Status: Estimated.

Claim: T-MAC achieves 6.6x kernel speedup and 11.1 tok/s on Raspberry Pi. Status: Confirmed.

Claim: FlashMoE beats LRU by 21% in hit rate. Status: Confirmed.

Claim: TIDE achieves 1.4-1.5x speedup with no accuracy drop. Status: Confirmed.

Claim: HERALD achieves near-lossless at 5-10% KV budget. Status: Confirmed.

Claim: KVDRIVE achieves 1.74x higher throughput. Status: Confirmed.

Claim: SmallThinker-21B-A3B achieves 30.19 tok/s on PC. Status: Confirmed.

Claim: The roofline paper confirms 64x arithmetic intensity difference. Status: Confirmed.

Claim: "Challenging GPU Dominance" shows CPU can outperform GPU on small models. Status: Confirmed.

Claim: BitMamba-2 released at 255M and 1B parameters. Status: Confirmed.

Claim: Camada combines 1.58-bit ternary with Sparse MoE. Status: Confirmed.

Claim: Ghidorah achieves 7.6x speedup on Jetson NX. Status: Confirmed.

Claim: ExpertFuse is hardware-agnostic and preserves expert contributions. Status: Confirmed.

Claim: storage.llm uses avg_gap_ema for predictive expert residency. Status: Confirmed.

Claim: AYOT achieves 8.97% gain over BitNet b1.58 2B4T on Math-500, GSM8K, HumanEval+, MBPP+. Status: Confirmed.

Claim: Maple Preview 20B-A1B runs at 28-34 tok/s on i5-8350U. Status: Confirmed.

Claim: Dell 7490 i7-8650U reaches 90°C and throttles to 30% sustained load. Status: Confirmed by Dell community.
Section 10 — Corrected Numbers and Constraints

The i7-8650U has AVX2 only. No AVX-512.

The Dell 7490 NVMe effective throughput is approximately 2 GB/s.

The Dell 7490 thermally throttles under sustained load. Reaches 90°C. Throttles at 80°C. Settles at 85°C with 30% sustained load.

Speculative decoding requires acceptance above 0.40 and draft compute ratio below 0.15. Auto-disable below 0.37.

Uniform 2-bit MoE quantization degrades quality. Use ternary or mixed 3-bit/4-bit with expert-specific allocation.

HiFC pSLC endurance benefit is shared with the OS and applications. pSLC is enterprise-tier only.

The realistic Tier 0 ceiling is 3 to 5 tok/s for 30B MoE and 3.5 to 4 tok/s for 7B ternary. The Maple Preview benchmark suggests these may be conservative.

q8_0 KV cache is sufficient for AT-3.

Activation quantization is required.

The expert-contiguous storage layout must be specified in the UMF container definition at Phase 0.

The predictive prerouter must be trained per model and shipped with the UMF container.

The bottleneck has shifted from SSD bandwidth to cache hit rate and prefetch accuracy.

The inter-step KV cache provides 1.6x speedup on diffusion.

The write-back ring buffer is 2-4 GiB DRAM staging with sequential flush.

The write budget for consumer drives is 100 GB/day. For enterprise, 1 TB/day.

SSMs (BitMamba-2, Camada) may be more efficient than transformers for Tier 0. Research Item 7 will resolve.

The arithmetic intensity difference between AR and diffusion is 64x at B=64. This is the formal justification for Pillar 1.

T-MAC is the AVX2 LUT kernel library for Tier 0.

FlashMoE is the ML-based cache replacement policy. 113KB per-layer model, 21% better than LRU.

TIDE is the interval-based expert refresh technique for MoE-dLLM.

HERALD is the diffusion KV offloading solution. 5-10% budget, near-lossless.

KVDRIVE is the SSD-aware KV layout. Semantic-Contiguity Packing plus Layer-Head Partitioning.

SmallThinker is a target model family for Tier 0.

AYOT is the ternarization method that makes post-training ternary conversion practical. 4M calibration tokens, 1M× fewer than training from scratch.

Maple Preview 20B-A1B runs at 28-34 tok/s on i5-8350U with TQ2_0 ternary and 5.5 GB file size.

Parable-Qwen3-4B (Q4_K_M) is the recommended coding model for near real-time feel on Tier 0. 2.5 GB quantized size.

Dell 7490 thermal behavior: 90°C under load, throttles to 30% sustained. This is a known model issue, documented in multiple Dell community threads.
Section 11 — Phase 0A Research Items

Phase 0A runs before any code is written. Eight research items. Each produces a written finding.

Research Item 1 — Diffusion on 4-Core CPU. Question: does diffusion language model inference beat autoregressive inference on a 4-core i7-8650U with DDR4-2400?

Specific techniques to benchmark: LLaDA-8B, Dream-7B, Eso-LMs, ADLM. Test on 4-core CPU. Measure against autoregressive baseline. The roofline paper predicts diffusion wins. The research question is empirical.

Research Item 2 — Ternary Model Quality. Question: is an 8B ternary model usable for real tasks?

Specific models: Ternary-Bonsai-8B, BitNet b1.58 2B4T, MAGNET/Genkidama, Trident, Camada, SmallThinker, AYOT-converted Qwen3-4B. Compare on MMLU, HellaSwag, ARC, WikiText-2 perplexity.

Research Item 3 — MoE Locality on PCIe 3.0. Question: can a 30B MoE run at 3 tok/s on PCIe 3.0 x4 NVMe with 25% expert cache residency?

Specific implementations: EDGE0, FlashMoE, FIRM-MoE, ExpertFuse, storage.llm, TIDE. Compare cache hit rate, throughput, accuracy.

Research Item 4 — KV Cache Quantization on CPU. Question: does q8_0 hold up at 64K context on a 16GB CPU-only machine?

Specific approaches: q8_0, q4_0, HERALD, KVDRIVE, RIS-Kernel, RaBitQCache. Compare perplexity impact, throughput, memory usage.

Research Item 5 — BitNet.cpp on AVX2-Only Hardware. Question: what is BitNet.cpp's actual speedup on AVX2-only CPUs?

Specific kernels: bitnet.cpp, T-MAC, sasori, ATLAS. Compare speedup, model quality, memory footprint.

Research Item 6 — Thermal Throttling Steady State. Question: can the Dell 7490 sustain 50% of initial throughput after 10 minutes?

Searches: sustained CPU frequency, effect of capping TDP, LLM benchmark on thermally constrained laptop. Dell community threads document 90°C and 30% sustained load.

Research Item 7 — SSM vs Transformer for Tier 0. Question: do BitMamba-2 or Camada at 1B parameters match or exceed transformer quality on Tier 0 tasks?

Specific models: BitMamba-2 255M, BitMamba-2 1B, Camada-Speculative, Camada-Consolidated. Compare against transformer baselines (SmallThinker-4B, Ternary-Bonsai-8B) on standard benchmarks.

If SSMs win, Pillar 1 shifts from diffusion-first to SSM-first. This is the highest-stakes open question.

Research Item 8 — Thermal Envelope Measurement. Question: what is the actual sustained throughput of the Dell 7490 across a 30-minute continuous inference run, and how does it vary with ambient temperature, battery state, and BIOS settings?

This feeds directly into Pillar 6.

Gate 0A: all eight research items produce written findings. No Phase 0B work begins until Gate 0A passes.
Section 12 — Implementation Roadmap

Phase 0B — Foundation. Weeks 1 through 8.

Deliverables: Repository with layout from specification. MIT license. llama.cpp and diffuse-cpp forked as submodules. Constraint-Aware Registry trait with conversion cost matrices, residency state, transfer latency, synchronization overhead, thermal state, thermal history field. SlyOS as design reference. Unified Model Format schema with expert-contiguous layout specification, KVDRIVE layout, thermal profile section, ML cache models. Hardware Detection Layer tested on Tier 0, 2, 3, 4. Bootstrap script.

Reading list: SlyOS, "Challenging GPU Dominance", CPU characterization paper, awesome-ai-cpu list, Ghidorah ARCA.

Gate 0B: CLI prints hardware profile as JSON on three tiers.

Phase 0C — Adaptive Performance Foundation. Weeks 9 through 12.

Deliverables: Performance Monitor. Performance Subclass determination. Live envelope tracking. Scheduler integration. Thermal history recording.

Gate 0C: The runtime detects thermal throttling on the Dell 7490 and automatically downgrades the execution plan without crashing or hanging.

Phase 1 — Core Engine. Weeks 13 through 28.

Deliverables: ATLAS TQ1.0 packer and inference engine. int4 FFN mode. T-MAC AVX2 LUT kernels. diffuse-cpp generation with CLAD cluster decoding and inter-step KV cache. Eso-LMs integration for exact diffusion KV caching. ADLM for anchored diffusion quality. AYOT ternarization pipeline. Prefill Optimizer. Automatic Mode Selector producing DAG-based execution plans. Ollama-compatible REST API.

Evaluate: SmallThinker as target model. Trident as Rust ternary reference. BitMamba-2 as SSM alternative. Parable-Qwen3-4B as coding model.

Gate 1: Tier 0 runs 7B ternary at 3.5 tok/s or higher. Tier 0 runs LLaDA-8B with CLAD at 6 tok/s or higher on factual prompts.

Phase 2 — Predictive MoE Streaming. Weeks 29 through 48.

Deliverables: I/O abstraction for io_uring, IOCP, kqueue. EDGE0 prerouter training pipeline. Expert-contiguous UMF storage layout. Block-level prefetching with arena allocator. FlashMoE ML cache replacement (113KB per-layer model shipped in UMF). FIRM-MoE fine-grained expert decomposition. ExpertFuse model-level expert merging. TIDE interval-based expert refresh. storage.llm avg_gap_ema reference. AdaptiveSD speculative decoding. MV-WSA RAM rebalancing.

Gate 2: Tier 0 runs Qwen3-30B-A3B at 3 tok/s or higher with 85% cache hit rate. Tail latency measured, not just mean.

Phase 3 — KV Cache and Memory. Weeks 49 through 56.

Deliverables: q8_0 KV cache. q4_0 fallback. HERALD for diffusion KV offloading. KVDRIVE SSD-aware layout. RIS-Kernel for long-context sparse attention. RaBitQCache evaluation. HiFC pSLC for enterprise only. Write budget enforcement. DRAM write-back ring buffer for pSLC tiers.

Gate 3: Tier 0 runs Llama-3.1-8B at 64K context under 10GB peak RAM.

Phase 4 — Hardware Abstraction. Weeks 57 through 72.

Deliverables: Backends for cpu (complete), cuda, metal, vulkan, rocm, webgpu, openvino, qnn, coreml, wasm. Per-layer DAG routing. SiliconScavenger-style concurrent scheduling. WASM fallback.

Gate 4: Same binary runs on Tier 0 through Tier 6 with automatic backend selection.

Phase 5 — Productization. Weeks 73 through 88.

Deliverables: Installer for Windows, macOS, Linux, Android, iOS. Model Manager with SmallThinker, Camada, BitMamba-2, Ternary-Bonsai, Maple Preview, Parable-Qwen3-4B in curated catalog. Desktop App. Python bindings. Update mechanism with rollback. PolyLink TIQE verification layer.

Gate 5: Non-technical user on Tier 0 installs, downloads a model, and has a conversation in under 5 minutes.

Phase 6 — Go-to-Market. Weeks 89 through 104.

Deliverables: Public releases. Community launch. Benchmark report. Enterprise offering. Governance.

Gate 6: 100K installs in year one.
Section 13 — Acceptance Tests

AT-1. Tier 0 ternary baseline. Falcon3-7B TQ1.0 or SmallThinker-4B at 3.5 tok/s or higher. Peak RAM under 12GB. WikiText-2 perplexity within 1% of FP16.

AT-2. Tier 0 MoE streaming. Qwen3-30B-A3B with int4 experts, TIDE interval refresh, FlashMoE cache policy at 3 tok/s or higher with 85% cache hit rate. Peak RAM under 15GB. Tail latency measured.

AT-3. Tier 0 long context. Llama-3.1-8B at q8_

---

## Section 14B — Extended Engineering Rules

Rule KILN-E1. No heredocs for code. Heredocs are permitted only for markdown documents.

Rule KILN-E2. Prefer one-liners where a one-liner is clearer than a multi-line block.

Rule KILN-E3. Files longer than roughly 50 lines are edited in the editor, not generated inline.

Rule KILN-E4. Read the file before modifying it. Every time.

Rule KILN-E5. Run a comprehensive diagnostic before every change.

Rule KILN-E6. Test counts come from pytest output, not from arithmetic.

Rule KILN-E7. One session equals one coherent unit of work.

Rule KILN-E8. Run git status --short before git commit. Run it before git checkout as well.

Rule KILN-E9. Every session ends with a docs update.

Rule KILN-E10. The repository is the source of truth, not working memory.

Rule KILN-E11. Command first, then code, in two separate blocks.

Rule KILN-E12. When a patch corrupts one file, fix that one file. Do not revert collateral damage elsewhere.

Rule KILN-E13. Anchor asserts prove the patch landed, not that the result is coherent. Verify the emitted code after every patch.

Rule KILN-E14. Any binary blob or serialized tensor in a patch must be one logical unit. No multi-line tensor literals.

Rule KILN-E15. When a fix breaks more than it lands, revert and defer.

Rule KILN-E16. Every deferred item gets a rationale and an estimate, or it does not get deferred.

Rule KILN-E17. Design first, code after. When a change touches more than two files or introduces a new abstraction, write the design doc first.

Rule KILN-E18. One session equals one commit. Not per-step commits.

Rule KILN-E19. Never paste tokens or credentials into chat. Rotate any that leak.

Rule KILN-E20. When a file has both a code path and a config path, they get separate diagnostics.

Rule KILN-E21. Speed-run mode is opt-in and honest about scope. When scope is too big for one session, cut to what is achievable and mark the rest deferred.

Rule KILN-E22. The repository is not a scratchpad. Personal notes, research files, and tool candidates go on the Desktop, not in docs/.

Rule KILN-E23. Test-local fakes must accept new kwargs as soon as the production signature changes. Same mechanical change across test files.

Rule KILN-E24. When the test count drops, stop and diagnose before committing.

Rule KILN-E25. After every session, update all docs before the push. README, CHANGELOG, interfaces, and any doc the session touched. The push is gated on docs being current.

Rule KILN-E26. Batch diagnostics with fixes. One read gathers every diagnostic. One fix per file. One commit closes it. Do not serialize into read-report-wait-fix-wait cycles.

Rule KILN-E27. One command per fix. Diagnose, then emit exactly one corrective command. No alternatives. No hedges.

Rule KILN-E28. Nothing loads that is not in the curated model catalog with a verified checksum.

Rule KILN-E29. Determinism means: same seed and same hardware state produce identical output.

Rule KILN-E30. The human is always in charge. Optional features are opt-in. No unattended behavior.

Rule KILN-E31. Show everything. When a command runs, its full output must be visible. Do not hide progress behind silent loops. Do not summarize the run after the fact without also showing the raw log. The user sees what the machine sees.

Rule KILN-E32. Group commands whenever possible. Serial commands that can be chained into one invocation must be chained. One shell block that does read-check-run-log-verify is preferred over four separate blocks. Time spent waiting between serialized steps is wasted. The user's time is the scarce resource.

Rule KILN-E33. Every long-running command writes its output to a log file AND streams it to the terminal. The log is for after. The stream is for now. Both are required.
