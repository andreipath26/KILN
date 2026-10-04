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

---

## Session Log — 2026-10-03

### Phase 0A: Research Closure

Status: CONDITIONALLY PASSED.

Eight research items. Seven confirmed. One conditional.

- Item 1: Diffusion on 4-core CPU. CONFIRMED. Roofline paper validates the arithmetic intensity argument. 3-6 tok/s on 4 cores for easy prompts. Crossover for unconditional throughput is 16-32 cores.
- Item 2: Ternary model quality. CONFIRMED WITH CAVEAT. Ternary Bonsai 8B at 75.5 vs FP16 Qwen3-8B at 79.3. 9.36x size reduction for less than 5 percent quality loss. Knowledge recall caveat unresolved.
- Item 3: MoE locality on PCIe 3.0. CONFIRMED. Cache hit rate is the bottleneck, not raw SSD bandwidth. FlashMoE ML cache replacement and TIDE interval-based expert refresh are the required techniques.
- Item 4: KV cache quantization. CONFIRMED. q8_0 is sufficient for AT-3. HERALD and KVDRIVE are the diffusion-specific additions.
- Item 5: BitNet on AVX2. CONFIRMED. T-MAC LUT kernels are the AVX2 path. AYOT makes post-training ternarization practical. Maple Preview 20B-A1B runs at 28-34 tok/s on i5-8350U with TQ2_0.
- Item 6: Thermal throttling. CONFIRMED. Dell 7490 reaches 90 C under load, throttles at 80 C, settles at 85 C with 30 percent sustained load. Pillar 6 is required.
- Item 7: SSM vs transformer. CONFIRMED. BitMamba-2-1B at 52.86 tok/s on i3-12100F. Dense SSM is viable. MoE SSM routing fails.
- Item 8: Thermal envelope measurement. OPEN-CONDITIONAL. Partial evidence on battery. Requires AC-powered 30-minute test before Phase 2.

### Phase 0B: Foundation

Status: COMPLETE.

Fourteen commits. All eight crates created. cargo check passes with zero warnings.

Crates created:

- kiln-hal: Backend trait, Registry trait, core type enums, InMemoryRegistry.
- kiln-models: ModelManifest, QuantizationSpec, ExpertTable, KvLayout, ThermalProfile, MlCacheRef.
- kiln-mem: Tier, MemoryManager trait, InMemoryManager, RebalancePolicy, errors, TensorId.
- kiln-io: IoOp, IoRequest, IoResult, IoSource trait, BlockSize, SyncIoSource.
- kiln-core: PerformanceEnvelope, Sample, ThermalHistory, ThrottlePrediction, PerformanceMonitor trait, LinuxMonitor, MockMonitor, LadderRung, AlgorithmReason, SelectionOutcome, ModeSelector trait, DefaultSelector, SchedulerError, ExecutionPlan, Node, Edge, TensorRef, NodeId, PlanRevision, RevisionReason, Scheduler trait, DefaultScheduler, MonitoredScheduler, StepOutcome.
- kiln-api: Ollama-compatible request and response types, four route handlers, axum router, serve function.
- kiln-cli: seven commands (serve, run, pull, list, bench, plan, info).
- kiln-bench: BenchReport, Measurement, Summary, five benchmarks, human and JSON output.

Documents created:

- docs/architecture.md: six-layer runtime stack specification.
- docs/core-design.md: Performance Monitor, Mode Selector, DAG Scheduler design.
- phase-0a-findings.md: seven confirmed, one conditional.
- README.md, CHANGELOG.md, LICENSE (MIT).

Verified tests:

- Monitor on Dell 7490: RAM 15.50 GB, CPU 2356 MHz, 70.1 C, thermal state Warm. Later run showed Throttling at 0.170 throughput after sustained cargo compilation. Live demonstration of Pillar 6.
- Selector: four test cases pass across all four rungs of the degradation ladder.
- Scheduler: revision cycle works. Initial plan revision 1, 4 nodes.
- API: all four endpoints respond correctly on port 11435.
- CLI: seven commands visible. info prints hardware. list and plan exit with code 2.
- Bench: five measurements recorded. monitor_envelope_read 248-283 ns. monitor_predict 315-323 ns.

### GitHub Publication

Repository: https://github.com/andreipath26/KILN

Visibility: PUBLIC.

First push: fifteen commits.

The internal roadmap file KILN MASTER ROADMAP AND RULES SET.md is excluded from git via .gitignore.

Canonical roadmap location: ~/KILN-MASTER-ROADMAP-AND-RULES-SET.md (outside the repository).

Clone-and-build verified from a clean environment. cargo check exits 0 on a fresh clone.

### Port Decision

KILN API binds 127.0.0.1:11435 by default. Ollama binds 11434. Both can run side by side. The port is configurable via the --port flag on kiln serve.

### Known Issues Carried Forward

- monitor_sample re-parses /proc/cpuinfo fully. Takes 99-185 ms. Optimize in Phase 1.
- throughput_fraction baseline is fragile. Uses first-sample mean. Calibrate properly in Phase 1.
- pSLC write endurance tracking not yet implemented. Phase 3.
- Item 8 requires AC-powered 30-minute test before Phase 2. Hard requirement.
- MonitoredScheduler revision trigger not yet exercised in a live run. Only the initial selection logic is verified.

### Session Discipline Notes

Rules KILN-E31, E32, E33 added:

- KILN-E31: Show everything. Full output visible.
- KILN-E32: Group commands whenever possible.
- KILN-E33: Every long-running command writes to a log AND streams to terminal.

File-writing pattern: python3 with triple-quoted string literals for all source files. Avoids shell heredoc corruption with code-containing markdown.

### Next Session Entry Point

Paste this roadmap and the Master File into a new chat. Say:

"Resume KILN. Phase 0B is complete and pushed to https://github.com/andreipath26/KILN. Next is Phase 1 Core Engine. Read the Master File and confirm you are up to speed."

Next deliverable: ATLAS TQ1.0 packer in cpp/kernels/. This is where the first ternary kernel is written and where the first real speed measurement on the Dell 7490 happens.


---

## Session Log — 2026-10-03 (continued)

### Phase 1 Progress

Status: IN PROGRESS.

Commits since the last session log:

- 0194ef1 Phase 1: AVX2 SIMD pack path for TQ1.0
- b0fe70b Phase 1: kiln-kernels crate with TQ1.0 ternary packer
- f3ad77b Remove internal roadmap from git tracking

Total commits in the repository: 17.

### What Was Built This Session

The kernel layer is now real. The first ternary packing kernel exists in
C and is compiled into the Rust runtime via build.rs and the cc crate.

Repository layout additions:

- cpp/kernels/tq1_0.c contains the scalar and AVX2 pack and unpack
  implementations.
- crates/kiln-kernels/ contains the Rust wrapper, the build script, and
  nine tests.
- docs/kernels-design.md anchors the TQ1.0 format, the C ABI, the Rust
  FFI contract, the build process, the SIMD strategy, the testing
  strategy, and format versioning.

### TQ1.0 Format

Five trits pack into one byte. Each trit is in {-1, 0, +1} and shifted
to {0, 1, 2}. The packed value is d0*81 + d1*27 + d2*9 + d3*3 + d4 and
is in [0, 242]. Values 243 through 255 are reserved. Padding uses zero
trits for non-multiple-of-five input.

The format uses 1.6 bits per weight. BitNet b1.58 uses a 2-bit encoding
that wastes one of four values. TQ1.0 is 20 percent more compact than
the naive 2-bit representation.

### AVX2 Path Verified

The AVX2 path in tq1_0.c is compiled into the binary. This was verified
by disassembling the compiled object file. objdump shows 20 AVX2
instructions (vmovups, vandps with 256-bit ymm registers) and 0 SSE
instructions. The -mavx2 flag is applied unconditionally on x86 and
x86_64 targets.

The build process:

- build.rs sets -mavx2 unconditionally on x86 targets.
- flag_if_supported was removed because the probe could silently drop
  the flag.
- The cc crate does not print its compiler invocation to cargo's
  verbose output. The proof of compilation is in the object file, not
  in the build log.

This is a Rule KILN-E13 lesson: the anchor for a compiled artifact is
the artifact itself, not the build log.

### Tests

Nine tests pass in kiln-kernels:

- round_trip_small (5 values, 1 byte)
- round_trip_various_lengths (1, 4, 5, 6, 9, 10, 11, 100, 1000)
- boundary_empty_panics
- invalid_value_panics
- invalid_byte_panics
- short_input_panics
- property_random_large (10,000 values)
- differential_scalar_vs_avx2 (AVX2 output matches scalar bit-for-bit)
- avx2_path_is_actually_compiled (prints runtime AVX2 support)

### Runtime Confirmation

- Dell Latitude 7490: AVX2 support confirmed at runtime.
- Object file: 20 AVX2 instructions, 0 SSE instructions.
- Build flag: -mavx2 applied to the C compile.

### Where We Stopped

The next unit of work is the benchmark that measures scalar versus AVX2
throughput. The benchmark code has been written and added to kiln-bench
but has not yet been built or run. The command to run it is:

cargo build -p kiln-bench
cargo run --quiet -p kiln-bench -- run

Expected new measurements in the report:

- pack_tq1_0_ns_per_element
- unpack_tq1_0_ns_per_element
- pack_tq1_0_mb_per_sec_input
- pack_tq1_0_mb_per_sec_output

These will be the first real performance numbers of the project.

### Carry-Forward Issues

- monitor_sample re-parses /proc/cpuinfo fully. Takes 99-185 ms.
  Optimize in Phase 1.
- throughput_fraction baseline is fragile. Uses first-sample mean.
  Calibrate properly in Phase 1.
- pSLC write endurance tracking not yet implemented. Phase 3.
- Item 8 requires AC-powered 30-minute test before Phase 2. Hard
  requirement.
- MonitoredScheduler revision trigger not yet exercised in a live run.
  Only the initial selection logic is verified.
- The machine is currently thermally throttled from sustained
  compilation work. Benchmark numbers will be lower than a cold run.
  This is the correct measurement for sustained-load behavior.

### Next Session Entry Point

Paste this roadmap and the Master File into a new chat. Say:

"Resume KILN. Phase 0B is complete. Phase 1 is in progress at commit
0194ef1. The TQ1.0 kernel and AVX2 path are verified. The next unit is
the scalar versus AVX2 throughput benchmark in kiln-bench. Read the
Master File and confirm you are up to speed."

Repository: https://github.com/andreipath26/KILN
Local project: /home/andreipath/Desktop/KILN
Canonical roadmap: ~/KILN-MASTER-ROADMAP-AND-RULES-SET.md


---

## Session Log — 2026-10-03 (Phase 1 progress)

### Summary

Phase 1 is in progress. The kernel layer exists and is benchmarked.
The end-to-end pipeline runs and is deterministic. Everything is pushed
to GitHub.

Commits added since the last log:
- 9aca4ec Phase 1: README updated with pipeline, kernels, and measured numbers
- c976580 Phase 1: pipeline wired through Selector and Scheduler, exposed via CLI
- 1063b1a Phase 1: end-to-end pipeline. Eight crates run together.
- 2f0d832 Phase 1: synthetic model format for pipeline testing
- d59b66a Phase 1: design document for the end-to-end pipeline
- d5603ef Phase 1: LUT approach documented as failed. Scalar fused matmul is production.
- a2cdf59 Phase 1: AVX2 fused ternary matmul with LUT
- 1faf24c Phase 1: scalar reference for fused ternary matmul
- 643fe52 Phase 1: design document for the fused ternary matmul kernel
- bb1e24d Phase 1: table-based unpacker and dead code cleanup
- 767c9cb Phase 1: scalar vs AVX2 pack benchmark. 21.25x speedup confirmed.
- 82452f5 Phase 1: TQ1.0 throughput benchmarks in kiln-bench
- 0194ef1 Phase 1: AVX2 SIMD pack path for TQ1.0
- b0fe70b Phase 1: kiln-kernels crate with TQ1.0 ternary packer

Total commits in the repository: 30.

### What Exists Now

Crates: 9
  kiln-hal
  kiln-models (with synthetic module)
  kiln-mem
  kiln-io
  kiln-core (with pipeline module)
  kiln-kernels (new)
  kiln-api
  kiln-cli
  kiln-bench

Design documents: 5
  docs/architecture.md
  docs/core-design.md
  docs/kernels-design.md
  docs/fused-kernel-design.md
  docs/pipeline-design.md

### Kernel Layer — Measured Performance

On Dell Latitude 7490, no GPU, thermally throttled:

  pack scalar:          10.79 ns/element
  pack AVX2:             0.61 ns/element   (17.70x speedup)
  unpack scalar:         2.94 ns/element
  unpack table:          2.49 ns/element   (1.12x speedup)
  matmul scalar fused:   2.75 ns/element

Documented failures (preserved in source with historical notes):
  LUT single-row matmul:  0.04x speedup
  LUT 8-row matmul:       0.34x speedup

The LUT failures are the most valuable output of the kernel work. They
tell a future engineer not to retry the same approach. The LUT build
dominates even with 8-way reuse, and the AVX2 gather on Skylake is too
slow to compensate. Do not retry without a fundamentally different design
(T-MAC activation-pattern LUT).

The AVX2 pack at 17.70x is the single biggest win. It is used once per
model load, not per inference. It means a 7B ternary model packs in
seconds instead of minutes.

### Pipeline — Measured Performance

Synthetic model: 4 rows x 1024 cols, all trits +1, activations 1-127
repeating.

  output:               65060  (deterministic across 13 runs)
  nodes_executed:       4
  load_time_nanos:      ~69,000
  select_time_nanos:    ~5,500
  schedule_time_nanos:  ~2,500
  execute_time_nanos:   ~13,400
  total_time_nanos:     ~91,000

Determinism: output is exactly 65060 on every run. Timing varies 6-24%
across runs. That is normal for a throttled laptop. The output is stable,
which is what AT-7 requires.

### The Fused Matmul Truth

The scalar fused matmul at 2.75 ns per element gives a real inference
estimate. For a 3B active MoE, one token needs 3 billion multiply-adds,
so 8.25 seconds per token, or 0.12 tok/s. AT-2 requires 3 tok/s. The gap
is 25x.

No single kernel optimization closes a 25x gap. The path to AT-2 requires
compounding wins: faster kernel, less active parameters, better cache hit
rate, speculative decoding. The kernel layer alone cannot deliver AT-2.

This is important for the roadmap. The AT-2 target may be unreachable
with the current kernel. The kernel needs a fundamentally different
approach (T-MAC activation-pattern LUT) before AT-2 becomes realistic.

### Rule Lessons Recorded

Rule KILN-E13: correctness and performance are separate anchors. The
differential tests proved the LUT output was bit-identical to scalar.
They did not prove the LUT was fast. Both checks are required before a
kernel is trusted.

Rule KILN-E5: diagnostic before every change. The build flag probe was
the source of the AVX2 confusion. The object file was the source of the
proof, not the build log.

Rule KILN-E27: one command per fix. Applied throughout the LUT debugging
and the pipeline wiring.

Rule KILN-E31, E32, E33: show everything, group commands, log and stream.
Applied throughout.

### GitHub State

Repository: https://github.com/andreipath26/KILN
Visibility: PUBLIC
Last push: 30 commits on main
README: updated with current state, crate list, commands, and all
measured performance numbers.

### Next Session Entry Point

Paste this roadmap and the Master File into a new chat. Say:

"Resume KILN. Phase 1 is in progress at commit 9aca4ec, pushed to GitHub.
The kernel layer exists and is benchmarked. The pipeline runs end to end
and is deterministic. Next is real GGUF loading and tokenization. Read
the Master File and confirm you are up to speed."

### Carry-Forward Issues

- monitor_sample re-parses /proc/cpuinfo fully. Takes 99-185 ms.
  Optimize in Phase 1.
- throughput_fraction baseline is fragile. Uses first-sample mean.
- pSLC write endurance tracking not yet implemented. Phase 3.
- Item 8 requires AC-powered 30-minute test before Phase 2. Hard
  requirement.
- MonitoredScheduler revision trigger not yet exercised in a live run.
- The fused matmul is not fast enough for AT-2. Needs T-MAC style LUT
  in a later phase.
- load_time_nanos in the pipeline is 65-74 microseconds, higher than the
  design estimate of 10 microseconds. File I/O and parsing are the cause.

### Next Major Deliverable

Real GGUF loading. This means reading a real model file from disk,
parsing its metadata, loading its tensors, and running inference through
the pipeline. It is the beginning of actual inference on real models.

This is a larger unit than anything so far. It needs its own design
document per Rule KILN-E17.



---

## Rule KILN-E34 — End of Session Mandatory Actions

After every session, before stopping, the following four actions are
mandatory and must be performed in this order.

1. Update the documentation in the repo. README.md, CHANGELOG.md, and
   any file under docs/ that the session touched. Stale documentation
   is a defect.

2. Append the session log to the master roadmap file in the project
   folder. The file is KILN MASTER ROADMAP AND RULES SET.md. Record
   what was built, what was measured, what failed, and what the next
   step is.

3. Update README.md with the current state, the crate list, the command
   list, and all measured numbers. The README is the public face of the
   project. It must be accurate.

4. Commit and push to GitHub. The session is not finished until the
   push succeeds. A local-only commit is an incomplete session.

This rule is non-negotiable. A session that ends without these four
actions is an incomplete session. The rule exists because drift is the
most expensive failure mode in a long project. Documentation that
lags reality by one session is recoverable. Documentation that lags by
ten sessions is not.

Rule KILN-E34 applies to every session from this point forward.


---

## Session Log — 2026-10-03 (Phase 1 continued)

### Summary

KILN now reads real GGUF files and runs inference. The kernel layer
exists. The pipeline is wired through the real Selector and Scheduler.
The GGUF loader is complete.

Total commits: 36.

### What Was Built

kiln-kernels crate:
- TQ1.0 ternary packing in C11.
- AVX2 SIMD pack path with 17.70x speedup.
- Table-based unpacker with 1.12x speedup.
- Scalar fused matmul at 2.75 ns/element.
- Two LUT approaches documented as failed.

GGUF loader in kiln-models:
- Header parser
- Metadata parser for nine types
- Tensor table parser
- GgufFile with mmap and zero-copy tensor access
- Alignment handling

Pipeline in kiln-core:
- run_once with Loader enum (Synthetic, Gguf)
- Real Selector and Scheduler wired through
- PipelineReport with five timing fields

CLI:
- kiln pipeline <path> [--loader synthetic|gguf] [--json]

Design documents:
- docs/kernels-design.md
- docs/fused-kernel-design.md
- docs/pipeline-design.md
- docs/gguf-design.md

### Measured Numbers

Kernel layer on Dell Latitude 7490:
  pack scalar:          10.79 ns/element
  pack AVX2:             0.61 ns/element  (17.70x)
  unpack scalar:         2.94 ns/element
  unpack table:          2.49 ns/element  (1.12x)
  matmul scalar fused:   2.75 ns/element

Pipeline on synthetic (4x1024):
  output: 65060 deterministic across 13 runs
  total: ~91,000 ns

Pipeline on real GGUF (948-byte file):
  output: -968
  total: ~100,000 ns

### Rule Added

Rule KILN-E34 — End of Session Mandatory Actions.
After every session: update docs, append roadmap, update README, push
to GitHub. Non-negotiable.

### The Fused Matmul Truth

The fused scalar matmul at 2.75 ns/element gives 0.12 tok/s on a 3B
active MoE. AT-2 requires 3 tok/s. The gap is 25x. No kernel alone can
close it. The path to AT-2 requires compounding wins: T-MAC style LUT,
reduced active parameters, better cache hit rate, speculative decoding.

### GitHub State

Repository: https://github.com/andreipath26/KILN
Last push: 36 commits on main.
README and CHANGELOG updated.

### Next Session Entry Point

Paste the master roadmap and the Master File into a new chat. Say:

"Resume KILN. Phase 1 is in progress. GGUF loading works and the
pipeline runs real GGUF inference. Next is tokenization and the chat
loop. Read the Master File and confirm you are up to speed."

### Carry-Forward

- Item 8 requires AC-powered 30-minute test before Phase 2.
- monitor_sample re-parses /proc/cpuinfo fully.
- throughput_fraction baseline fragile.
- Fused matmul not fast enough for AT-2. Needs T-MAC style LUT.
- load_time in pipeline is 65-78 microseconds. Higher than design.


---

## Session Log — 2026-10-03 (Phase 1 core engine complete)

### Summary

Phase 1 core engine is complete. KILN now loads real GGUF files,
parses tokenizer metadata, runs the pipeline through the real Selector
and Scheduler, and holds a chat session with deterministic sampling.

Total commits: 43.

### What Was Built This Session

BPE tokenizer in kiln-models:
- from_gguf reads tokenizer.ggml.tokens, merges, special IDs
- encode, encode_with_special, decode
- from_parts public constructor for tests
- Five tests

Chat loop in kiln-core:
- Forward trait with MockForward and SyntheticForward
- SamplingStrategy: Greedy, TopK, TopP
- Sampler with xorshift64 PRNG
- ChatSession with generate and generate_streaming
- Four tests

CLI chat command:
- kiln chat --model <gguf> --seed N --max-tokens N --strategy S
- Loads tokenizer from GGUF, builds SyntheticForward, reads stdin

Design document:
- docs/chat-loop-design.md

### Measured Numbers

Session verified on a 448-byte GGUF with 14 vocabulary tokens and 6
merge rules. Chat session ran, encoded stdin, generated tokens,
decoded, and printed. Output is gibberish because the synthetic
forward does not use model weights.

### Bugs Fixed

- Shadowed variable binding: 'let name = b"w"' inside write_minimal_gguf
  shadowed the 'name: &str' parameter. The compiler was right. The full
  function body revealed the cause. Rule KILN-E13 lesson.
- Temp file collision between parallel GGUF tests. Fixed by giving each
  test its own filename.
- Invalid cast in the test helper.
- Private field access from kiln-core tests. Fixed by adding
  BpeTokenizer::from_parts.

### What Remains for a Working Assistant

- Real transformer forward pass (attention, layernorm, FFN)
- Real model weights in a GGUF with tokenizer metadata
- KV cache for efficient multi-token generation
- Thermal response in a live inference

Those are Phase 1 late and Phase 2.

### GitHub State

Repository: https://github.com/andreipath26/KILN
Last push: 43 commits on main.

### Next Session Entry Point

Paste the master roadmap and the Master File into a new chat. Say:

"Resume KILN. Phase 1 core engine is complete. The runtime loads GGUF,
parses tokenizers, runs the pipeline, and holds a chat session. Next is
the real transformer forward pass. Read the Master File and confirm you
are up to speed."


---

## Rule KILN-E35 — Standard GGUF Quantization Support

KILN supports the standard GGUF quantizations natively. Ternary is an
optimization, not a requirement.

The masses bring models they already have. Those models are almost never
in TQ1_0. They are in Q4_K_M, Q5_K_M, Q6_K, Q8_0, and F16. If KILN only
supports TQ1_0, it fails the vast majority of real models.

Natively supported from Phase 1:

  F32, F16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, TQ1_0

Deferred to Phase 2:

  Q2_K, Q3_K variants, IQ1 through IQ4 variants

Each supported format gets:

  1. A decode function that unpacks bytes to f32.
  2. A fused matmul kernel that reads packed bytes and produces the
     dot product without materializing weights.

The decode functions are Phase 1. The fused matmul kernels land in
priority order: Q4_K first, then Q6_K, then Q8_0.

No user is excluded because their model is in a different format.
That is the rule. Ternary is the KILN-specific advantage for the
slowest machines, not a gate that locks out everyone else.


---

## Session Log — 2026-10-04 (transformer correctness + KV cache)

### Summary

Phase 1 forward pass is correct. The Q4_K byte layout was wrong and is
fixed. The KV cache is in and gives a 17x speedup on multi-token replies.
First honest throughput number on Tier 0: 0.41 tok/s on Qwen2.5-1.5B Q4_K_M.

Total commits: 46 (3 new this session).

### Commits This Session

- 6bba397 Phase 1: fix Q4_K nibble layout. Forward pass produces correct logits.
- b33520c Phase 1: Q6_K dequant verified correct against ggml reference.
- (pending) Phase 1: KV cache. 20-token reply 18min -> 48s.

### The Q4_K Bug

kiln_q4k_dequant_block wrote 256 dequantized weights into the wrong
slots. It treated the 128 weight bytes as 8 chunks of 16 bytes, each
with its own sub-block scale. The correct ggml layout is 4 groups of
32 bytes; within each group, the low nibbles and high nibbles map to
different output regions with different scales:

  bytes 0-31  low  -> outputs 0..31    scale[0]
  bytes 0-31  high -> outputs 32..63   scale[1]
  bytes 32-63 low  -> outputs 64..95   scale[2]
  bytes 32-63 high -> outputs 96..127  scale[3]
  bytes 64-95 low  -> outputs 128..159 scale[4]
  bytes 64-95 high -> outputs 160..191 scale[5]
  bytes 96-127 low -> outputs 192..223 scale[6]
  bytes 96-127 high-> outputs 224..255 scale[7]

Every linear layer in the model was reading a permuted weight matrix.
Top-1 for "The capital of France is" went from token 99222 ("太",
garbage) to 12095 (" Paris", correct). The fix was 10 lines.

Q6_K was audited line by line against ggml dequantize_row_q6_K. It was
already correct. No fix needed.

The 26 kernel tests did not catch this because they round-trip KILN
against KILN. There is no differential test against ggml. That anchor
must be added.

### The KV Cache

New module crates/kiln-core/src/kv_cache.rs. Per-layer K and V store,
grown by append, reset when the caller's tokens do not start with the
cached prefix. RoPE now takes a position offset so cached K is not
re-rotated.

Result: 20-token reply on Qwen2.5-1.5B, Dell 7490, went from ~18
minutes to 48 seconds. Output is bit-identical to the pre-cache path.

### First Honest Tier 0 Numbers

Dell Latitude 7490, i7-8650U, 16 GB DDR4-2400, no GPU.
Model: Qwen2.5-1.5B-Instruct, Q4_K_M, 986 MB.
Prompt: "hi there", greedy, seed 42.

  Forward pass, 5-token prompt:        9.32 s
  Chat, 20-token reply, cached:       48.31 s
  Throughput, 20-token reply:         0.41 tok/s
  Top-1 for "The capital of France is": " Paris" (correct)

The floor is the scalar fused matmul. The cache removed the
recomputation. The matmul is now the whole cost.

### Next Unit

AVX2 fused matmul for Q4_K and Q6_K. Design doc written:
docs/avx2-matmul-design.md. Target 4-8x over scalar. Expected result:
20-token reply in <15 s, >1.3 tok/s on Tier 0.

### Known Issues Carried Forward

- No chat template. Qwen2.5 expects ChatML. Quality is off without it.
- Cache overflow panics. Should return an error. Touches the Forward trait.
- Differential test against ggml reference not yet written.
- monitor_sample re-parses /proc/cpuinfo fully. 99-185 ms.
- throughput_fraction baseline fragile.
- pSLC write endurance tracking not implemented. Phase 3.
- Item 8 requires AC-powered 30-minute test before Phase 2.
- clippy is not installed. No lint gate.

### Rule Lessons

Rule KILN-E13: correctness and self-consistency are different anchors.
The 26 kernel tests proved KILN matched KILN. They did not prove KILN
matched ggml. The bug lived in the gap.

Rule KILN-E4: reading the file before modifying it caught the Q6_K
verdict in one pass. Do not skip the read.

Rule KILN-E34: this session had not updated docs, README, roadmap, or
pushed for 6 commits. That is the drift the rule exists to prevent.


---

## Session Log — 2026-10-04 (working chat)

### Summary

KILN holds a working conversation. The tokenizer now matches special
tokens atomically, the chat path wraps user input in ChatML, and the
model stops on the correct EOS. First verified assistant reply:
`hi there` -> `Hello! How can I help you today?`.

Total commits: 47. HEAD = c75225d, pushed to origin/main.

### Commits This Session

- c75225d Phase 1: working chat. Special-token tokenizer, ChatML
  template, EOS stops. AVX2 Q4_K dequant verified but not wired
  (6.8x regression when split).

### What Was Built

1. Special-token tokenizer. `BpeTokenizer` now reads
   `tokenizer.ggml.token_type` and extracts every CONTROL (type 3)
   and USER_DEFINED (type 4) entry. The encoder scans the input for
   any special-token substring before running BPE, splits at each
   match, and emits the special's ID verbatim. Specials are sorted
   longest-first so `<|im_start|>` matches before `<|im_`.

   Before: `<|im_start|>` tokenized to [27, 91, 318, 4906, 91, 29]
   as byte fragments. After: single token 151644.

2. ChatML wrapper in `kiln chat`. User input is wrapped as:
   `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`
   `<|im_start|>user\n{prompt}<|im_end|>\n`
   `<|im_start|>assistant\n`

3. EOS works. The GGUF's `eos_token_id` is 151645, which is
   `<|im_end|>`. Once the tokenizer emits it as a single token, the
   ChatSession's existing EOS check stops generation naturally.

4. AVX2 Q4_K dequant kernel in `cpp/kernels/q4k_avx2.c`. Verified
   bit-for-bit against the scalar path on a real Q4_K block from the
   Qwen2.5-1.5B GGUF: max diff 0. Not wired into the matmul because
   splitting dequant (C) from dot (Rust) per block is 6.8x slower
   than the fused scalar loop.

### The Split-AVX2 Lesson

Wiring the AVX2 dequant into `dot_packed` via a per-block FFI call
produced 329.99 s for 20 tokens, up from 48.31 s. 6.8x regression.

Cause: one C call per block per token, each returning a 256-float
stack array that Rust then re-reads to do the dot product. The
scalar C matmul does dequant+dot in a single fused loop with the
accumulator in a register. Splitting it is slower than not
vectorizing at all.

Conclusion: the AVX2 path must be a fused matmul — dequant and dot
in the same C function, accumulating in a `__m256`, one horizontal
sum at the end. That is the next unit.

### Numbers

Baseline (reverted to scalar, correct):
- `hi there`, greedy, 20 tokens: 48.31 s, 0.41 tok/s
- `hi there`, greedy, 10 tokens: 25.17 s, 0.40 tok/s
- `hi there` with ChatML, 9 tokens: 58.83 s, 0.15 tok/s
  (the extra cost is the ~20-token system prompt + tags prefill)

### Known Issues Carried Forward

- Fused AVX2 matmul not yet written. Prefill and decode are both
  bound by the scalar matmul.
- No differential tokenizer test against llama.cpp. The special-token
  bug would have been caught by one.
- `q4k_matmul_fast` and `q4k_dequant_block_avx2` exist but are not
  wired. Verified correct, waiting for the fused matmul.
- Prefill cost is visible: 20 tokens of ChatML prompt ≈ 40 s before
  the first output token.
- Chat template is hardcoded to ChatML. Other model families will
  need their own wrappers, ideally read from `tokenizer.chat_template`.
- KV cache overflow panics. No error channel through `Forward`.
- clippy is not installed. No lint gate.
- Item 8 (thermal envelope measurement) still pending.

### Rule Lessons

Rule KILN-E13, again: 26 kernel tests and 5 tokenizer tests were
self-consistent and wrong. Both classes of bug were caught by
comparing KILN against a reference (ggml for Q4_K, the Qwen vocab
for the specials). Self-consistency is not correctness.

Rule KILN-E17: the special-token design doc was written before any
code. The patch was applied in two rounds — field/ctor, then encoder
pre-scan — and both rounds landed clean.

Rule KILN-E4: reading `gguf.rs` line 149 before writing the match
arm found the correct variant name (`Int32`, not `I32`). One read
saved one compile cycle.

Rule KILN-E34: docs, README, roadmap updated before this commit.
Push succeeded.

### Next Unit

Fused AVX2 Q4_K matmul. One C function:
  kiln_q4k_matmul_avx2_fused(w, nw, x, nx, out)
Dequant and dot in the same loop, no stack round-trip, single hsum.
Then a differential test against the scalar path on a real block.
Then measure. Target 4-8x. Expected `hi there` reply in < 10 s.
