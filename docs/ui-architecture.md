# KILN UI architecture and continuation guide

**Status:** The local chat and model-library UI is shipped in Linux preview packages. Automated browser and desktop-window checks pass; validation on a real Kali desktop and hardware-tier qualification remain open.
**Roadmap authority:** [KILN MASTER ROADMAP AND RULES SET](../KILN%20MASTER%20ROADMAP%20AND%20RULES%20SET.md). This document translates that roadmap into UI responsibilities; it does not change phase order, implementation status, or acceptance targets.

## 1. Product intent

The KILN UI is the local-first control surface for the whole KILN runtime. It must cover the roadmap's model lifecycle, heterogeneous execution, unified memory, performance and quality controls, and future workload types without becoming a CPU-only laptop dashboard or a chat-only product.

The workspace has four top-level destinations: **Chat**, **Models**, **Activity** and **System details**. Chat is the default simple experience; System details exposes the settings and capabilities the active runtime actually supports. All destinations use the same model, run, configuration and API state. They must not create separate inference paths or separate setting stores. System details adds progressive disclosure for supported controls rather than a second product. Recent conversation selection and deletion must remain available on narrow screens as well as desktop.

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

### Chat mode

- **Setup and model library:** assess the machine and discover local GGUFs on dashboard open. If no model is active, automatically try up to three candidates, preferring the largest structurally valid model estimated to fit the resident budget, then verified experimental CPU-streamable MoE candidates. Recheck every decision during activation and show failures; never present model size or GGUF structure as backend/performance certification. Remote search/download stays explicit and requires user selection.
- **Task launch:** present workload-oriented choices rather than low-level backend names. The first delivered path is text generation/chat; additional workload panels appear as their runtime APIs land.
- **Inference workspace:** streamed output, stop, retry/regenerate, selected model, generation progress, and locally stored conversation history with clear retention/clear controls.
- **Automatic operation:** choose sensible model and runtime defaults without requiring CLI knowledge. Show the selected model/runtime policy, what KILN actually activated, and any fallback. Allow a user to open Advanced mode at the same task/model/run.

### System details

Panels are registered by capability and grouped by task. Roadmap areas that must fit this shell include:

- **Model and precision:** GGUF and later UMF/sidecar artifacts, quantization policy, TQ1_0, model provenance and quality evidence.
- **Device/backend capabilities:** detected devices, operations, supported precisions, memory topology, transfer/conversion costs, and fallback coverage.
- **Execution plan:** per-operation/per-layer placement and the reason for each decision. Allow explicit overrides only where supported and safe.
- **Memory hierarchy:** RAM, VRAM/unified memory, NVMe, KV cache, expert cache/residency, admission budgets and measured usage without double-counting shared physical memory.
- **Scheduling and performance:** thread/batch settings, prefill/decode metrics, thermal state, I/O overlap, expert prediction/prefetch and replacement policy as each becomes real.
- **Quality and diagnostics:** doctor, canary/watchdog, exact/approximate mode, output-quality checks, backend errors, logs and reproducible run reports.
- **Future workloads:** diffusion and other roadmap tasks plug into the same shell and capability contract; do not force their controls into a text-chat-specific state model.

System details should use progressive disclosure within each panel. “Everything configurable” means every supported setting is discoverable, validated and operable; it does not mean every knob is shown simultaneously or that unsupported settings are fabricated.

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

The current dashboard bootstrap also reports a local startup recommendation based on live resident-memory headroom and verified model layout. When no session is active, the browser tries a bounded set of candidates through the existing activation API; activation performs the authoritative compatibility and resource checks. This is a CPU/RAM heuristic only. It does not automatically fetch weights, discover accelerators, or guarantee model quality/performance.

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
- Keep conversations and profiles local with clear delete/export controls. Conversation backups are user-initiated and clearly identified as containing chat text and model identifiers that may include local paths. Diagnostic bundles are opt-in and redacted.
- Validate every API setting and model path server-side. Constrain paths to approved model roots where appropriate, reject traversal, enforce download limits and verify hashes before atomic install.
- Keep model/license/source information visible before download. Gated artifacts must follow provider authorization and must not be bypassed.
- Preserve fail-safe CPU fallback and make fallbacks visible. Never let stale browser state imply that an unsupported device or setting is active.
- UI failure must not break CLI or API inference. No external CDN/assets are required for the local UI to start.

## 7. Packaging and frontend organization

The UI is a local web client served by the KILN API inside a native GTK desktop window on Linux. The window uses the system WebKitGTK engine and loads only KILN's loopback origin; users do not need a general-purpose browser to manage KILN. The `.deb` declares the Python GObject and WebKitGTK runtime packages so Kali/Debian installs bring in the window dependencies. The tar archive remains a one-executable release; its installer checks for the system GTK/WebKit runtime and creates a desktop launcher only when it is available. Without those optional desktop dependencies, the CLI binary still installs and the installer prints the Kali/Debian package command needed to enable the desktop window. The API remains independently usable by Ollama-compatible clients and `kiln serve`. Keep the client boundary replaceable so Phase 8 can change the desktop shell without rewriting runtime contracts. Static assets are bundled with the API; ordinary use does not require a separately running frontend server. The exact frontend toolchain may evolve, but retain:

- a small application shell and four-destination navigation;
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
6. Add a System details overview showing only live capabilities/settings; do not imply that MoE graph streaming, per-layer backend placement, diffusion or unimplemented controls work.

### Expand by roadmap capability

- Phase 3.5: reproducible dense settings, benchmark runs and comparisons.
- Phase 4/4.5: quantization/quality status and doctor/canary/watchdog results.
- Phase 5: model-sidecar status, expert residency/I/O/prefetch, exact-router guard state, cache and end-to-end telemetry only after runtime integration.
- Phase 6: diffusion task panels when an actual runtime path exists.
- Phase 7: device/backend/operation capability panels and safe placement overrides.
- Phase 8: packaging, desktop wrapper, accessibility, support, upgrade and release qualification.

### Acceptance

- Chat and System details operate on the same session and settings; navigation loses no work and changes no runtime behavior.
- UI is usable offline for local models and inference; no external assets or service are required.
- UI/API/CLI agree on model inventory and model-management behavior.
- Capability state and accepted/active settings match the server; unsupported operations fail clearly and safely.
- Model downloads show provenance/license, progress, cancellation and verification before activation.
- Streaming, stop/cancel, model switch and server error states are coherent; no stale model/session state is shown.
- UI reports measured vs estimated values distinctly and includes the experiment identity needed to reproduce a benchmark.
- Keyboard navigation, screen-reader labels, contrast, narrow-window layout and large text work.
- Full roadmap capabilities enter through extension points without a new UI architecture or per-hardware-mode rewrite.

## 9. Current implementation status and handoff

The UI serves bundled assets from the existing Axum process. Linux `kiln dashboard` opens those assets inside a GTK/WebKit app window, restricts top-level navigation to the local KILN origin, and stops the API instance it started when the window closes. The `.deb` installs the GTK/WebKit Python bindings and app icon. The redesigned workspace has Chat, Models, Activity and System details navigation; live memory/headroom figures and model-fit state come from the existing API, and the Activity page stores action metadata locally without prompts or generated text. It provides local conversation export, validated import/merge, and deletion controls separately from the prompt-free activity log. Backups include chat text and model identifiers that may contain local filesystem paths; merge refuses to discard histories beyond the per-model storage limit. On narrow screens, recent conversation selection and deletion remain available through an expandable history panel. Automatic model setup progress and outcomes appear in the visible system-check panel while Chat is the default destination, as well as in Models. Chat supports local GGUF inventory, explicit Hugging Face search and verified downloads, activation, streamed chat, stop, response regeneration and per-model conversation history. System details reports CPU/RAM defaults, active model, API-reported capability states and current inference behavior. The chat output limit (1–4096 tokens) is editable in System details and shared with chat; sampling remains greedy. The adapted JSONL parser is reused by chat and downloads, with strict malformed-event errors. The API binds only to loopback and adds a restrictive content security policy, no-sniff/frame protections, a no-referrer policy and no-store responses. CI verifies the redesigned browser UI, exercises the GTK/WebKit window under a virtual display, and installs the actual release-built `.deb` with apt to launch and close its registered desktop entry. Installation and use on a real Kali desktop remains the final environment-specific check.

**Automated release checks:** CI builds and starts the actual loopback API with no configured model, then drives the bundled UI in headless Chromium. Fixture-backed scenarios cover model search/download progress and state recovery; a separate scenario downloads a small public GGUF through KILN's own verified model library, loads it, switches between real GGUF paths, checks failed-load restoration, chats with the live model, and cancels another model transfer while checking that its partial file is removed. The suite also checks served-page security headers, no-model startup and visible automatic setup feedback from Chat, Chat/Models/Activity/System details keyboard navigation, axe-core serious/critical findings, accessible-name snapshots, narrow-window overflow, forced-colors keyboard focus and uncaught JavaScript errors. Rust API tests cover the actual server's no-model responses and loopback/security behavior. CI additionally installs the release-built Debian package through apt, launches its registered desktop entry under Xvfb, checks the loaded KILN window and API, and verifies both shut down when the window closes. A real Kali desktop installation and use remains unverified.

**Remaining deployment work:** the CI smoke model and fixture-backed browser scenarios qualify functionality, not performance or quality on each supported hardware tier. Before making tier-specific performance claims, run the KILN roadmap's acceptance workload on that tier and record the measured result. Runtime performance/quality telemetry, general profile management, execution planning, device discovery, and future workload panels depend on their runtime/API capabilities and are not prerequisites for this current local chat/model-library UI release. Keep the model manager API and client modular; do not copy transport/hash/file-walking logic into JavaScript or API handlers.

**Verification status:** the browser suite is in `tools/ui-e2e` and runs with the workspace test job. Its CI job installs Rust and Chromium, runs the actual API, fetches the Apache-2.0 SmolLM2-135M Q4_K_M smoke artifact from Hugging Face, and validates the transfer digest before inference. The local implementation workspace lacks Rust and browser binaries, so the hosted workflow is the execution environment for these checks; no separate frontend build is needed because assets are bundled at compile time.
