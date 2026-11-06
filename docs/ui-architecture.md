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

The existing API has `/api/chat`, `/api/generate`, `/api/tags` and `/api/version`. It loads at most one GGUF at process startup from `KILN_MODEL_PATH`; it does not yet expose a complete model manager, dynamic model activation, hardware capability inventory, execution plans, settings schema, or telemetry stream. The recently merged CLI `list`, `search`, `pull` and `run auto` behavior is not automatically an API/UI capability.

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

- Bind to loopback by default; remote access must require a separate explicit configuration and suitable authentication/origin protections before shipping.
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

The initial implementation now serves bundled `index.html`, `ui.css`, and `ui.js` from the existing Axum process. The Simple view streams chat through `/api/chat`; the Advanced view reads `/api/tags`, `/api/version`, and `/api/ui/capabilities` and reports the current CPU/RAM profile and active startup-loaded model. It also states the current inference defaults: greedy sampling, a fixed 128-token chat output limit, and role-prefixed prompt formatting. The capability snapshot explicitly marks UI model management, dynamic activation, accelerator discovery, per-layer placement, end-to-end MoE streaming, and diffusion as unavailable or not started. The UI stores chat history in browser local storage, scoped by model, and supports stopping an in-flight request.

**Remaining first-slice work:** connect model discovery, search, download, and activation through a shared model-management service before exposing those controls in the UI; add settings/session APIs and live performance/quality telemetry as the runtime supports them. Keep the model manager API and client modular; do not copy the CLI's curl/hash/file-walking logic into JavaScript or API handlers. Then add responsive/accessibility review and complete the verification gates.

**Verification status:** JavaScript syntax check and `git diff --check` pass. Rust compilation and UI behavior have not been verified in this environment. No dedicated UI test suite or frontend build is present yet. Later implementers should compile the Rust workspace, run API/UI tests, exercise offline startup and local-model chat, test model pull integrity/cancellation and activation failure, inspect keyboard/screen-reader behavior, and verify README/roadmap capability claims. Until those checks pass, label the UI as development-only and list the unchecked gates.
