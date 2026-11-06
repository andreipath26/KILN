# KILN UI architecture and continuation guide

**Status:** Architecture baseline plus initial local UI implementation; development-only until the verification gates below pass.
**Roadmap authority:** [KILN MASTER ROADMAP AND RULES SET](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md). This document translates that roadmap into UI responsibilities; it does not change phase order, implementation status, or acceptance targets.

## 1. Product intent

The KILN UI is the local-first control surface for the whole KILN runtime. It must cover the roadmap's model lifecycle, heterogeneous execution, unified memory, performance and quality controls, and future workload types without becoming a CPU-only laptop dashboard or a chat-only product.

The UI has two top-level modes:

1. **Simple** — find or obtain a suitable model, choose a supported task, and run it with sensible automatic settings.
2. **Advanced** — inspect and control every setting the active runtime/backend actually supports.

These modes are two views of the same model, run, configuration and API state. They must not create separate inference paths or separate setting stores. Simple mode is progressive disclosure over Advanced mode, not a second product.

## 2. Product constraints from the roadmap

- **Platform scope:** CPU, GPU, iGPU/APU, NPU, RAM, accelerator memory and NVMe where available. The Dell Latitude 7490 is the minimum acceptance machine, not the UI's product boundary.
- **Execution model:** one user-facing model/API path, hardware-aware per-layer dispatch, a replaceable backend registry and safe fallback. Do not encode a fixed “CPU mode”, “GPU mode” or list of vendor names as the architecture.
- **Local by default:** the application shell, model inventory, settings, conversations, diagnostics and inference remain usable without cloud connectivity. Remote model search/download is an explicit network action.
- **Truthful capability state:** distinguish available, selected, active, fallback, experimental, unsupported and not-yet-implemented. A GGUF parser accepting a file does not prove the inference backend can load its architecture.
- **Evidence:** show measured values separately from estimates, targets and external research results. A cache-hit rate or predictor score must not be displayed as end-to-end performance.
- **Quality and safety:** surface canary/watchdog status, exact versus approximate execution, failure reason, and recovery/fallback. Never hide a rejected operation or silently continue with invalid output.
- **Phase boundaries:** Phase 3.5 dense tuning, Phase 5 MoE streaming, Phase 6 diffusion, Phase 7 hardware abstraction and Phase 8 productization stay separate roadmap tracks. The UI can host all of them without claiming future capabilities are implemented.
- **One-source-of-truth:** UI actions call KILN's API/runtime contracts. Do not implement inference, model compatibility, placement decisions or memory accounting in JavaScript.

## 3. Information architecture

### Simple mode

- **Setup and model library:** discover local GGUFs, search a remote catalog, inspect model size/source/license/format, download with explicit consent, verify the artifact, and choose a model. Explain the difference between structural validation, backend load support, and hardware-tested support.
- **Task launch:** present workload-oriented choices rather than low-level backend names. The first delivered path is text generation/chat; additional workload panels appear as their runtime APIs land.
- **Inference workspace:** streamed output, stop, retry/regenerate, selected model, generation progress, and locally stored conversation history with clear retention/clear controls.
- **Automatic operation:** show the selected runtime policy, what KILN actually activated, and any fallback. Allow a user to open Advanced mode at the same task/model/run.

### Advanced mode

Panels are registered by capability and grouped by task. Roadmap areas that must fit this shell include:

- **Model and precision:** GGUF and later UMF/sidecar artifacts, quantization policy, TQ1_0, model provenance and quality evidence.
- **Device/backend capabilities:** detected devices, operations, supported precisions, memory topology, transfer/conversion costs, and fallback coverage.
- **Execution plan:** per-operation/per-layer placement and the reason for each decision. Allow explicit overrides only where supported and safe.
- **Memory hierarchy:** RAM, VRAM/unified memory, NVMe, KV cache, expert cache/residency, admission budgets and measured usage without double-counting shared physical memory.
- **Scheduling and performance:** thread/batch settings, prefill/decode metrics, thermal state, I/O overlap, expert prediction/prefetch and replacement policy as each becomes real.
- **Quality and diagnostics:** doctor, canary/watchdog, exact/approximate mode, output-quality checks, backend errors, logs and reproducible run reports.
- **Future workloads:** diffusion and other roadmap tasks plug into the same shell and capability contract; do not force their controls into a text-chat-specific state model.

Advanced mode should use progressive disclosure within each panel. “Everything configurable” means every supported setting is discoverable, validated and operable; it does not mean every knob is shown simultaneously or that unsupported settings are fabricated.

## 4. Capability-driven extension model

The UI must be data-driven from a versioned KILN capability snapshot. A device/backend descriptor should be able to report, as available:

- stable device/backend identifier and human-readable label;
- supported operations, workload types and precision/quantization formats;
- memory kinds/capacities, shared-memory relationships and transfer paths;
- known bandwidth/latency or conversion costs with provenance and confidence;
- thermal/power state and current health;
- operation-level constraints, fallback backend and whether support is experimental;
- the runtime controls and telemetry that this capability exposes.

The client consumes capabilities and renders registered panels/controls. It does not hard-code assumptions such as “GPU means CUDA” or treat a compiled backend as proof of runtime dispatch. Backend-specific extensions can add controls without changing the top-level navigation. Every control declares its type, bounds, default, active value, restart requirement, and support state; server validation remains authoritative.

Settings have **desired**, **accepted**, and **active** values. The UI shows the diff and only marks a value active after the runtime confirms it. Configuration is profile-scoped (task/model/device/runtime fingerprint); switching models or hardware cannot silently apply an incompatible profile.

## 5. UI/API boundary

The existing API has `/api/chat`, `/api/generate`, `/api/tags` and `/api/version`. The UI also has local model inventory, public Hugging Face GGUF search/download with progress and disconnect cancellation, and model activation endpoints backed by shared `kiln-models::library` operations. It still does not expose a complete device capability inventory, execution plans, a general settings schema, or runtime telemetry. The CLI `list`, `search` and `pull` behavior is shared through the model library service, rather than scraped from CLI output.

Build a versioned UI-facing API surface rather than making the browser scrape CLI output or duplicate its behavior. Initial contract areas:

| API area | Purpose |
|---|---|
| Bootstrap/capabilities | UI build/API version, runtime state, available tasks, devices, backends, controls and support states. |
| Model inventory | Local candidates, metadata/provenance, structural validation, backend load validation, install status and active model. |
| Model search/download | Explicit remote search, selected artifact download progress, hash verification, cancellation, and atomic install. |
| Runtime/session | Activate/deactivate a model, create/stop a task session, stream output, cancel, reset and report active settings. |
| Plan/configuration | Preview execution and memory plan; validate and apply profile changes with accepted/active diff. |
| Events/telemetry | Progress, device/resource changes, fallback, performance and quality events; bounded, local and attributable to a run. |
| Diagnostics | Doctor results, capability mismatches, actionable errors and opt-in redacted diagnostic export. |

Preserve the existing Ollama-compatible endpoints. UI-specific APIs must not break their contract. Long-running operations need asynchronous progress/cancel semantics; do not hold the inference mutex while downloading or probing devices. Local search/download must reuse a shared model-management service, not retain a CLI-private implementation.

## 6. Privacy, security and resilience

- Bind to loopback only; the current API rejects non-loopback hosts because remote access has no authentication or origin-protection design yet.
- Make remote catalog/search/download a clear network boundary. Never upload prompts, model metadata, diagnostics or telemetry by default.
- Keep conversations and profiles local with clear delete/export controls. Diagnostic bundles are opt-in and redacted.
- Validate every API setting and model path server-side. Constrain paths to approved model roots where appropriate, reject traversal, enforce download limits and verify hashes before atomic install.
- Keep model/license/source information visible before download. Gated artifacts must follow provider authorization and must not be bypassed.
- Preserve fail-safe CPU fallback and make fallbacks visible. Never let stale browser state imply that an unsupported device or setting is active.
- UI failure must not break CLI or API inference. No external CDN/assets are required for the local UI to start.

## 7. Packaging and frontend organization

The UI is a local web client served by the KILN API, with a replaceable client boundary so Phase 8 can package it as a desktop application without rewriting runtime contracts. Static assets should be versioned and shipped with the application; avoid requiring a separately running frontend server for ordinary use. The exact frontend toolchain may evolve, but retain:

- a small application shell and two-mode navigation;
- feature modules by domain (models, chat/tasks, devices, memory, execution, diagnostics);
- typed API/capability models and one centralized client;
- shared design tokens/components and accessible keyboard/focus behavior;
- no inference or resource-accounting logic in the client;
- versioned local storage migrations and a clear boundary between preferences, profiles and conversations.

### Reuse from Ollama

KILN should reuse compatible pieces from [Ollama's official app UI](https://github.com/ollama/ollama/tree/main/app/ui/app), rather than transplant its full desktop application. That frontend is a React/Vite application with an Ollama browser client, route/query infrastructure, model picker and streaming chat components. Its API module also calls app-specific user, cloud, integration and desktop endpoints; those contracts and the Go desktop server do not exist in KILN. Copying the whole app would add a large dependency surface and present controls KILN cannot support.

The first direct reuse is Ollama's JSONL streaming parser from `app/ui/app/src/util/jsonl-parsing.ts`, adapted into the bundled client at `crates/kiln-api/static/stream.js`. KILN keeps strict parse failures so malformed events fail visibly. Future UI work should reuse isolated components or patterns only when their contracts fit; preserve Ollama's MIT notice for copied/adapted source and document KILN-specific changes. The model picker, chat composer, stop behavior, keyboard interactions and download progress are implementation references, not proof that their app-only flows can be carried over unchanged.

## 8. Implementation sequence and acceptance gates

### UI foundation — first slice

1. Serve a bundled local UI from KILN's existing API process without relying on a cloud asset host.
2. Implement the two top-level modes and a responsive application shell.
3. Add a versioned capability/bootstrap endpoint and render capability/support states from it.
4. Implement model library inventory/search/pull/activation through shared server services; keep CLI and UI on the same model-management logic.
5. Implement a functional streaming chat path over the existing API, with local conversation state and explicit active model/status.
6. Add an Advanced overview showing only live capabilities/settings; do not imply that MoE graph streaming, per-layer backend placement, diffusion or unimplemented controls work.

### Expand by roadmap capability

- Phase 3.5: reproducible dense settings, benchmark runs and comparisons.
- Phase 4/4.5: quantization/quality status and doctor/canary/watchdog results.
- Phase 5: model-sidecar status, expert residency/I/O/prefetch, exact-router guard state, cache and end-to-end telemetry only after runtime integration.
- Phase 6: diffusion task panels when an actual runtime path exists.
- Phase 7: device/backend/operation capability panels and safe placement overrides.
- Phase 8: packaging, desktop wrapper, accessibility, support, upgrade and release qualification.

### Acceptance

- Both modes operate the same session and settings; switching mode loses no work and changes no runtime behavior.
- UI is usable offline for local models and inference; no external assets or service are required.
- UI/API/CLI agree on model inventory and model-management behavior.
- Capability state and accepted/active settings match the server; unsupported operations fail clearly and safely.
- Model downloads show provenance/license, progress, cancellation and verification before activation.
- Streaming, stop/cancel, model switch and server error states are coherent; no stale model/session state is shown.
- UI reports measured vs estimated values distinctly and includes the experiment identity needed to reproduce a benchmark.
- Keyboard navigation, screen-reader labels, contrast, narrow-window layout and large text work.
- Full roadmap capabilities enter through extension points without a new UI architecture or per-hardware-mode rewrite.

## 9. Current implementation status and handoff

The UI serves bundled assets from the existing Axum process. Simple mode supports local GGUF inventory, explicit Hugging Face search and verified downloads, activation, streamed chat, stop, response regeneration, and per-model browser-local conversation history. The Advanced view reports CPU/RAM defaults, active model, API-reported capability states, and current inference behavior. The chat output limit (1–4096 tokens) is editable in Advanced and shared with Simple chat; sampling remains greedy. The adapted JSONL parser is reused by chat and downloads, with strict malformed-event errors. The API binds only to loopback and adds a restrictive content security policy, no-sniff/frame protections, a no-referrer policy, and no-store responses.

**Remaining deployment work:** the workspace CI covers compilation and unit checks for bundled UI headers, loopback binding, and model-path validation. It does not run a real model or a browser. Before a supported release, exercise startup without a model, local-model chat, download cancellation/integrity failure, activation recovery, and responsive/keyboard/screen-reader behavior. Runtime performance/quality telemetry, general profile management, execution planning, device discovery, and future workload panels depend on their runtime/API capabilities and are not prerequisites for this current local chat/model-library UI release. Keep the model manager API and client modular; do not copy transport/hash/file-walking logic into JavaScript or API handlers.

**Verification status:** JavaScript syntax check and `git diff --check` pass. This checkout has no Rust toolchain; use the linked PR's latest workspace CI result as the Rust/API build gate. No separate frontend build is needed because assets are bundled at compile time. The local UI is usable from the source build, but release deployment readiness also requires the real-model, download, activation-recovery, and accessibility checks listed above.
