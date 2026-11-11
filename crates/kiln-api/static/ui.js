import { parseJsonlFromResponse } from "./stream.js";

const $ = (selector) => document.querySelector(selector);
const state = {
  model: null,
  messages: [],
  controller: null,
  busy: false,
  capabilities: null,
  models: [],
  remoteModels: [],
  modelActionBusy: false,
  downloadController: null,
  maxTokens: 128,
};

const tabs = [
  { button: $("#tab-simple"), panel: $("#panel-simple") },
  { button: $("#tab-advanced"), panel: $("#panel-advanced") },
];

function selectTab(index) {
  tabs.forEach((tab, i) => {
    const active = i === index;
    tab.button.classList.toggle("is-active", active);
    tab.button.setAttribute("aria-selected", String(active));
    tab.button.tabIndex = active ? 0 : -1;
    tab.panel.hidden = !active;
  });
  tabs[index].button.focus();
}

tabs.forEach((tab, index) => {
  tab.button.addEventListener("click", () => selectTab(index));
  tab.button.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      selectTab((index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length);
    }
  });
});

function historyKey() {
  return `kiln.ui.chat.v1:${state.model?.path || state.model?.name || "unconfigured"}`;
}

function settingsKey() {
  return `kiln.ui.settings.v1:${state.model?.path || state.model?.name || "unconfigured"}`;
}

function readSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey()) || "{}");
    const value = Number(saved.max_tokens);
    state.maxTokens = Number.isInteger(value) ? Math.max(1, Math.min(4096, value)) : 128;
  } catch {
    state.maxTokens = 128;
  }
  $("#max-tokens").value = String(state.maxTokens);
}

function saveSettings() {
  const value = Number($("#max-tokens").value);
  state.maxTokens = Number.isInteger(value) ? Math.max(1, Math.min(4096, value)) : 128;
  $("#max-tokens").value = String(state.maxTokens);
  try {
    localStorage.setItem(settingsKey(), JSON.stringify({ max_tokens: state.maxTokens }));
  } catch {
    $("#model-library-status").textContent = "Could not save this model's settings in browser storage.";
  }
}

function readHistory() {
  try {
    let stored = localStorage.getItem(historyKey());
    const legacyModelIsUnambiguous = state.model?.name
      && !state.models.some((model) => model.name === state.model.name && model.path !== state.model.path);
    if (stored === null && state.model?.path && legacyModelIsUnambiguous) {
      const legacyKey = `kiln.ui.chat.v1:${state.model.name}`;
      stored = localStorage.getItem(legacyKey);
      if (stored !== null) {
        try {
          localStorage.setItem(historyKey(), stored);
          localStorage.removeItem(legacyKey);
        } catch {
          // Retain and display the old conversation if browser storage is full.
        }
      }
    }
    const value = JSON.parse(stored || "[]");
    return Array.isArray(value) ? value.filter((item) =>
      item && ["user", "assistant"].includes(item.role) && typeof item.content === "string") : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(historyKey(), JSON.stringify(state.messages.slice(-100)));
  } catch {
    showChatError("Could not save this conversation in local browser storage.");
  }
}

function renderMessages() {
  const container = $("#messages");
  const shouldStickToBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
  container.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = '<div class="empty-glyph" aria-hidden="true">✳</div><strong>Your local chat starts here</strong><p>Send a message to begin. Your conversation is saved in this browser on this device.</p>';
    container.append(empty);
  } else {
    for (const [index, message] of state.messages.entries()) {
      const item = document.createElement("article");
      item.className = `message ${message.role}`;
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = message.role === "user" ? "You" : "KILN";
      const content = document.createElement("div");
      content.textContent = message.content;
      item.append(label, content);
      if (!state.busy && message.role === "assistant" && index === state.messages.length - 1
          && state.messages[index - 1]?.role === "user") {
        const actions = document.createElement("div");
        actions.className = "message-actions";
        const retry = document.createElement("button");
        retry.type = "button";
        retry.className = "button button-quiet";
        retry.textContent = "Regenerate";
        retry.setAttribute("aria-label", "Regenerate the last response");
        retry.addEventListener("click", () => void regenerateResponse());
        actions.append(retry);
        item.append(actions);
      }
      container.append(item);
    }
  }
  if (shouldStickToBottom) container.scrollTop = container.scrollHeight;
  const enabled = Boolean(state.model);
  $("#prompt").disabled = !enabled || state.busy;
  $("#send-button").disabled = !enabled || state.busy;
  $("#clear-chat").disabled = !enabled || !state.messages.length || state.busy;
  $("#chat-subtitle").textContent = state.model ? "Streaming from the local KILN API" : "Waiting for a loaded model";
}

function showChatError(message = "") {
  $("#chat-error").textContent = message;
}

function announceChat(message) {
  $("#chat-announcer").textContent = message;
}

function setServerState(online, label) {
  $("#server-dot").classList.toggle("is-online", online);
  $("#server-dot").classList.toggle("is-offline", !online);
  $("#server-label").textContent = label;
}

function renderSystemProfile(profile) {
  const list = $("#system-profile");
  list.replaceChildren();
  const stateBadge = $("#profile-state");
  stateBadge.textContent = profile ? "Reported" : "Unavailable";
  stateBadge.className = `badge ${profile ? "good" : "warn"}`;
  const rows = profile ? [
    ["Detected tier", profile.tier],
    ["CPU logical cores", profile.cpu_cores],
    ["Installed RAM", `${profile.ram_gb} GB`],
    ["Available RAM", Number.isFinite(profile.available_ram_bytes)
      ? `${(profile.available_ram_bytes / 1_073_741_824).toFixed(2)} GiB` : "Unavailable"],
    ["Estimated model residency budget", Number.isFinite(profile.resident_budget_bytes)
      ? `${(profile.resident_budget_bytes / 1_073_741_824).toFixed(2)} GiB` : "Unavailable"],
    ["Context default", profile.n_ctx],
    ["Threads default", profile.n_threads],
  ] : [["Profile", "Not available"]];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value ?? "—";
    row.append(term, detail);
    list.append(row);
  }
  $("#profile-note").textContent = profile
    ? "Available RAM and residency budget are live estimates for model selection, not guaranteed working-set limits. Accelerator discovery is not yet exposed by the API."
    : "The runtime did not return a system profile.";
}

function renderSession(model, bootstrap) {
  const list = $("#session-details");
  list.replaceChildren();
  const rows = [
    ["Model", model?.name || "None loaded"],
    ["Model size", model ? `${(model.size / 1_073_741_824).toFixed(2)} GiB` : "—"],
    ["Session config", model ? "Active in KILN runtime" : "Not configured"],
    ["Inference mode", model ? (
      bootstrap?.active_model?.mode === "experimental_cpu_moe_stream"
        ? "Experimental CPU expert streaming (large model)"
        : "Standard / mmap model loading"
    ) : "—"],
    ["Native expert cache", bootstrap?.active_model?.mode === "experimental_cpu_moe_stream"
      && Number.isFinite(bootstrap?.active_model?.expert_cache_mib)
      ? `${bootstrap.active_model.expert_cache_mib} MiB (memory-budgeted)` : "—"],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    row.append(term, detail);
    list.append(row);
  }
  const status = $("#session-state");
  status.textContent = model ? "Active" : "No model";
  status.className = `badge ${model ? "good" : "warn"}`;
  $("#active-model-name").textContent = model?.name || "No model loaded";
  $("#active-model-chip").title = model?.name || "Configure KILN_MODEL_PATH before starting the API";
  $("#no-model-notice").hidden = Boolean(model);
  $("#app-version").textContent = bootstrap?.api_version ? `API ${bootstrap.api_version}` : "development build";
}

function renderLocalModels(models) {
  state.models = Array.isArray(models) ? models : [];
  const list = $("#local-models");
  list.replaceChildren();
  if (!state.models.length) {
    const empty = document.createElement("p");
    empty.className = "muted small";
    empty.textContent = "No structurally valid local GGUF files were found in the standard model locations.";
    list.append(empty);
    return;
  }
  for (const model of state.models) {
    const row = document.createElement("article");
    row.className = "model-entry";
    const main = document.createElement("div");
    main.className = "model-entry-main";
    const title = document.createElement("div");
    title.className = "model-entry-title";
    title.textContent = model.name;
    const meta = document.createElement("div");
    meta.className = "model-entry-meta";
    meta.textContent = `${(model.size_bytes / 1_073_741_824).toFixed(2)} GiB · ${model.source}${model.active ? " · Active" : ""}`;
    main.append(title, meta);
    const aliases = Array.isArray(model.aliases)
      ? model.aliases.filter((name) => typeof name === "string" && name !== model.name)
      : [];
    if (aliases.length) {
      const alternate = document.createElement("div");
      alternate.className = "model-entry-meta";
      const shown = aliases.slice(0, 6);
      alternate.textContent = `Also installed as: ${shown.join(", ")}${aliases.length > shown.length ? ` (+${aliases.length - shown.length} more)` : ""}`;
      main.append(alternate);
    }
    if (model.load_error) {
      const warning = document.createElement("div");
      warning.className = "model-entry-meta";
      warning.setAttribute("role", "status");
      warning.textContent = model.load_error;
      main.append(warning);
    }
    const actions = document.createElement("div");
    actions.className = "model-entry-actions";
    const choose = document.createElement("button");
    choose.type = "button";
    choose.className = model.active ? "button button-quiet" : "button button-secondary";
    choose.textContent = model.active ? "Active model" : "Use model";
    choose.disabled = Boolean(model.load_error) || model.active || state.modelActionBusy || state.busy;
    if (model.load_error) choose.title = model.load_error;
    choose.addEventListener("click", () => void activateLocalModel(model.path, "auto"));
    actions.append(choose);
    // KILN-owned Hugging Face model downloads know their source repository.
    // Repair only absent split files; the existing large shard is never
    // downloaded again or overwritten.
    if (model.load_error && Array.isArray(model.repair_missing)) {
      for (const item of model.repair_missing.slice(0, 4)) {
        const repair = document.createElement("button");
        repair.type = "button";
        repair.className = "button button-secondary";
        repair.textContent = "Download missing shard";
        repair.title = item.file;
        repair.disabled = state.modelActionBusy || state.busy;
        repair.addEventListener("click", () => void downloadArtifact(item.repository, item.file));
        actions.append(repair);
      }
    }
    if (!model.active && !model.load_error && model.native_stream_candidate === true) {
      const stream = document.createElement("button");
      stream.type = "button";
      stream.className = "button button-secondary";
      stream.textContent = "Try streaming";
      stream.title = "Experimental selected-expert CPU reads for this verified fused MoE layout (single- or multi-shard); runtime and performance remain unverified";
      stream.disabled = state.modelActionBusy || state.busy;
      stream.addEventListener("click", () => void activateLocalModel(model.path, "stream"));
      actions.append(stream);
    } else if (!model.active && !model.load_error && model.native_stream_candidate === false) {
      const reason = document.createElement("span");
      reason.className = "muted small";
      reason.textContent = "Native expert streaming not supported for this GGUF layout";
      reason.title = "Ordinary mmap or lazy expert loading may still be available";
      actions.append(reason);
    }
    row.append(main, actions);
    list.append(row);
  }
}

// Group exact conventional GGUF shard names within one remote repository.
// Never mix quantizations, subdirectories or similarly named model families.
function splitArtifactSelection(artifacts, selected) {
  const match = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/i.exec(selected);
  if (!match) return { files: [selected], split: false, missing: [], error: "" };
  const count = Number(match[3]);
  if (!Number.isSafeInteger(count) || count < 2 || count > 1024) {
    return { files: [], split: true, missing: [],
      error: "Invalid GGUF split count; cannot download a partial model." };
  }
  const expected = Array.from({ length: count }, (_, index) =>
    `${match[1]}-${String(index + 1).padStart(5, "0")}-of-${match[3]}.gguf`);
  const available = new Set(artifacts.map((item) => item.path));
  const missing = expected.filter((file) => !available.has(file));
  return {
    files: missing.length ? [] : expected, split: true, missing,
    error: missing.length
      ? `Incomplete remote split GGUF: ${missing.length} of ${count} shards are not listed. KILN will not download a partial model.`
      : "",
  };
}

function renderRemoteModels(models) {
  state.remoteModels = Array.isArray(models) ? models : [];
  const list = $("#remote-models");
  list.replaceChildren();
  if (!state.remoteModels.length) {
    const empty = document.createElement("p");
    empty.className = "muted small";
    empty.textContent = "No public GGUF repositories matched this search.";
    list.append(empty);
    return;
  }
  for (const model of state.remoteModels) {
    const row = document.createElement("article");
    row.className = "model-entry";
    const main = document.createElement("div");
    main.className = "model-entry-main";
    const title = document.createElement("div");
    title.className = "model-entry-title";
    title.textContent = model.repository;
    const meta = document.createElement("div");
    meta.className = "model-entry-meta";
    meta.textContent = `${model.license} · ${Number(model.downloads || 0).toLocaleString()} downloads${model.gated ? " · gated repository" : ""}`;
    main.append(title, meta);
    const actions = document.createElement("div");
    actions.className = "model-entry-actions";
    for (const artifact of model.artifacts || []) {
      const pull = document.createElement("button");
      pull.type = "button";
      pull.className = "button button-secondary";
      const size = artifact.size_bytes ? ` · ${(artifact.size_bytes / 1_073_741_824).toFixed(2)} GiB` : "";
      pull.textContent = `Download ${artifact.path}${size}`;
      pull.disabled = Boolean(model.gated) || state.modelActionBusy;
      pull.title = model.gated ? "This repository requires provider authorization and cannot be downloaded through this public flow." : "Download and verify this GGUF file";
      const selection = splitArtifactSelection(model.artifacts || [], artifact.path);
      if (selection.split) {
        // Show one action even if shard 1 is absent from the provider's list,
        // so an incomplete repository is visibly diagnosed rather than hidden.
        const sameModel = /^(.+)-\d{5}-of-(\d{5})\.gguf$/i.exec(artifact.path);
        const firstListed = (model.artifacts || [])
          .map((item) => item.path)
          .filter((name) => {
            const other = /^(.+)-\d{5}-of-(\d{5})\.gguf$/i.exec(name);
            return other && sameModel && other[1] === sameModel[1] && other[2] === sameModel[2];
          })
          .sort()[0];
        if (artifact.path !== firstListed) continue;
        if (selection.error) {
          pull.textContent = "Incomplete split model (download unavailable)";
          pull.disabled = true;
          pull.title = selection.error + " Check the provider repository for the missing parts.";
        } else {
          pull.textContent = `Download complete model (${selection.files.length} shards)`;
          pull.title = `Download and individually SHA-256 verify all ${selection.files.length} original GGUF shards`;
        }
      }
      if (!selection.error) {
        pull.addEventListener("click", () =>
          void downloadArtifact(model.repository, artifact.path, selection.files));
      }
      actions.append(pull);
    }
    if (model.gated) {
      const note = document.createElement("span");
      note.className = "muted small";
      note.textContent = "Provider authorization required";
      actions.append(note);
    } else if (!model.artifacts?.length) {
      const note = document.createElement("span");
      note.className = "muted small";
      note.textContent = "No GGUF files listed";
      actions.append(note);
    }
    row.append(main, actions);
    list.append(row);
  }
}

async function loadLocalModels() {
  const response = await fetch("/api/ui/models", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not scan local models.");
  const payload = await response.json();
  renderLocalModels(payload.models || []);
}

async function activateLocalModel(path, mode = "auto") {
  if (state.modelActionBusy) return;
  state.modelActionBusy = true;
  setModelActionState(true, "Loading selected model…");
  try {
    const response = await fetch("/api/ui/models/activate", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ path, mode }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Model activation failed.");
    await refresh();
    setModelActionState(false, `Active model: ${payload.model}`);
  } catch (error) {
    await refresh();
    setModelActionState(false, error.message || "Model activation failed.");
  }
}

async function searchModels(query) {
  if (state.modelActionBusy || !query.trim()) return;
  state.modelActionBusy = true;
  setModelActionState(true, "Searching Hugging Face for public GGUF models…");
  $("#model-search-button").disabled = true;
  try {
    const url = `/api/ui/models/search?q=${encodeURIComponent(query.trim())}&limit=10`;
    const response = await fetch(url, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Model search failed.");
    renderRemoteModels(payload.models || []);
    setModelActionState(false, `${payload.models?.length || 0} repositories found.`);
  } catch (error) {
    setModelActionState(false, error.message || "Model search failed.");
  } finally {
    $("#model-search-button").disabled = false;
  }
}

async function downloadArtifact(repository, file, requestedFiles = [file]) {
  if (state.modelActionBusy) return;
  const files = [...new Set(requestedFiles)];
  if (!files.length || files.length > 1024) {
    setModelActionState(false, "Invalid split-model download selection.");
    return;
  }
  state.modelActionBusy = true;
  state.downloadController = new AbortController();
  const signal = state.downloadController.signal;
  const modelLabel = files.length > 1 ? `${files.length}-shard GGUF model` : file;
  setModelActionState(true, `Starting download: ${modelLabel}`, true);
  let installedCount = 0;
  try {
    for (const [index, currentFile] of files.entries()) {
      if (signal.aborted) throw new DOMException("Download cancelled", "AbortError");
      const response = await fetch("/api/ui/models/pull", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ repository, file: currentFile }),
        signal,
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.error || `Model download request failed (${response.status}).`);
      }
      let installedPath = null;
      for await (const event of parseJsonlFromResponse(response)) {
        if (event.event === "progress") {
          const received = Number(event.bytes_received || 0);
          const total = Number(event.total_bytes || 0);
          const amount = `${(received / 1_073_741_824).toFixed(2)} GiB`;
          const pct = total > 0 ? ` (${Math.min(100, Math.floor(received * 100 / total))}%)` : "";
          const prefix = files.length > 1 ? `Shard ${index + 1}/${files.length} · ` : "";
          $("#model-library-status").textContent = `${prefix}Downloading ${currentFile}: ${amount}${pct}`;
        } else if (event.event === "done") installedPath = event.path;
        else if (event.event === "error") throw new Error(event.error || "Model download failed.");
      }
      if (!installedPath) {
        throw new Error("The server closed the download stream before confirming installation.");
      }
      installedCount += 1;
    }
    await loadLocalModels();
    setModelActionState(false, files.length === 1
      ? `Verified and installed ${file}. Select it from local models to load it.`
      : `Verified ${installedCount}/${files.length} files for ${modelLabel}. Select the complete model from local models.`);
  } catch (error) {
    const message = error.name === "AbortError"
      ? (files.length === 1 ? "Download canceled; the partial file was removed."
        : `Download canceled after ${installedCount}/${files.length} verified files. Completed shards were preserved; the partial download was removed.`)
      : `Download stopped after ${installedCount}/${files.length} verified files: ${error.message || "unknown error"}. Completed shards were preserved.`;
    await loadLocalModels().catch(() => {});
    setModelActionState(false, message);
  } finally {
    state.downloadController = null;
  }
}

function setModelActionState(busy, message, cancellable = false) {
  state.modelActionBusy = busy;
  $("#model-library-status").textContent = message;
  $("#model-search-button").disabled = busy;
  $("#refresh-models").disabled = busy;
  $("#cancel-download").hidden = !busy || !cancellable;
  renderLocalModels(state.models);
  renderRemoteModels(state.remoteModels);
}

function humanState(value) {
  return String(value || "unknown").replaceAll("_", " ");
}

function renderCapabilities(bootstrap) {
  state.capabilities = bootstrap;
  $("#capability-version").textContent = `Schema ${bootstrap.schema_version ?? "—"}`;
  const body = $("#capability-rows");
  body.replaceChildren();
  for (const capability of bootstrap.capabilities || []) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const status = document.createElement("td");
    const detail = document.createElement("td");
    const badge = document.createElement("span");
    name.textContent = capability.label;
    badge.className = `capability-state ${String(capability.status).replaceAll("_", "-")}`;
    badge.textContent = humanState(capability.status);
    status.append(badge);
    detail.textContent = capability.detail;
    row.append(name, status, detail);
    body.append(row);
  }
  if (!body.children.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.className = "muted";
    cell.textContent = "No capability information was returned.";
    row.append(cell);
    body.append(row);
  }
  renderSystemProfile(bootstrap.system_profile);
}

async function refresh() {
  try {
    const [versionResponse, tagsResponse, capabilitiesResponse, modelsResponse] = await Promise.all([
      fetch("/api/version", { cache: "no-store" }),
      fetch("/api/tags", { cache: "no-store" }),
      fetch("/api/ui/capabilities", { cache: "no-store" }),
      fetch("/api/ui/models", { cache: "no-store" }),
    ]);
    if (![versionResponse, tagsResponse, capabilitiesResponse, modelsResponse].every((response) => response.ok)) {
      throw new Error("KILN API returned an error while loading the local runtime state.");
    }
    const [version, tags, capabilities, models] = await Promise.all([
      versionResponse.json(), tagsResponse.json(), capabilitiesResponse.json(), modelsResponse.json(),
    ]);
    const localModels = models.models || [];
    const activeModelPath = localModels.find((model) => model.active)?.path;
    state.model = tags.models?.[0] ? { ...tags.models[0], path: activeModelPath } : null;
    setServerState(true, "Local runtime connected");
    renderSession(state.model, {
      api_version: version.version,
      active_model: capabilities.active_model,
    });
    renderCapabilities(capabilities);
    renderLocalModels(localModels);
    readSettings();
    state.messages = readHistory();
    renderMessages();
  } catch (error) {
    state.model = null;
    state.messages = [];
    readSettings();
    setServerState(false, "Local runtime unavailable");
    renderSession(null, null);
    renderLocalModels([]);
    $("#profile-state").textContent = "Unavailable";
    $("#profile-state").className = "badge warn";
    $("#session-details").lastElementChild.lastElementChild.textContent = "Could not reach API";
    $("#capability-rows").innerHTML = '<tr><td colspan="3" class="muted">Start or reconnect to the KILN API to load capabilities.</td></tr>';
    renderMessages();
    showChatError(error.message || "Could not connect to KILN.");
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("#send-button").textContent = busy ? "Stop" : "Send ↗";
  $("#send-button").classList.toggle("button-primary", !busy);
  $("#send-button").classList.toggle("button-secondary", busy);
  $("#prompt").disabled = busy || !state.model;
  $("#send-button").disabled = !busy && !state.model;
  $("#clear-chat").disabled = busy || !state.messages.length;
}

async function sendMessage(text) {
  if (!state.model || state.busy || !text.trim()) return;
  showChatError();
  state.messages.push({ role: "user", content: text.trim() });
  const requestMessages = state.messages.map(({ role, content }) => ({ role, content }));
  const answer = { role: "assistant", content: "" };
  state.messages.push(answer);
  renderMessages();
  await runAssistant(requestMessages, answer);
}

async function regenerateResponse() {
  if (!state.model || state.busy || state.messages.length < 2) return;
  const answerIndex = state.messages.length - 1;
  if (state.messages[answerIndex].role !== "assistant" || state.messages[answerIndex - 1]?.role !== "user") return;
  const requestMessages = state.messages.slice(0, answerIndex).map(({ role, content }) => ({ role, content }));
  const answer = state.messages[answerIndex];
  answer.content = "";
  renderMessages();
  await runAssistant(requestMessages, answer);
}

async function runAssistant(requestMessages, answer) {
  setBusy(true);
  announceChat("Generating response.");
  state.controller = new AbortController();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: state.model.name, messages: requestMessages, stream: true, options: { num_predict: state.maxTokens } }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.error || `Chat request failed (${response.status}).`);
    }
    for await (const event of parseJsonlFromResponse(response)) {
      consumeChatEvent(event, answer);
      renderMessages();
    }
    if (!answer.content && !state.controller.signal.aborted) {
      throw new Error("The model returned an empty response.");
    }
    saveHistory();
    announceChat("Response complete.");
  } catch (error) {
    if (error.name === "AbortError") {
      if (!answer.content) state.messages.pop();
      saveHistory();
      announceChat("Generation stopped.");
    } else {
      state.messages.pop();
      saveHistory();
      showChatError(error.message || "Inference failed.");
    }
  } finally {
    state.controller = null;
    setBusy(false);
    renderMessages();
    $("#prompt").focus();
  }
}

function consumeChatEvent(event, answer) {
  if (event.error) throw new Error(event.error);
  if (event.message?.content) answer.content += event.message.content;
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    state.controller?.abort();
    return;
  }
  const prompt = $("#prompt");
  const text = prompt.value;
  prompt.value = "";
  prompt.style.height = "auto";
  void sendMessage(text);
});

$("#prompt").addEventListener("input", (event) => {
  const textarea = event.currentTarget;
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
});

$("#prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.busy) $("#chat-form").requestSubmit();
  }
});

$("#clear-chat").addEventListener("click", () => {
  if (state.busy) return;
  state.messages = [];
  showChatError();
  try {
    localStorage.removeItem(historyKey());
    if (state.model?.path && state.model?.name
        && !state.models.some((model) => model.name === state.model.name && model.path !== state.model.path)) {
      localStorage.removeItem(`kiln.ui.chat.v1:${state.model.name}`);
    }
  } catch {
    showChatError("The conversation was cleared from view, but browser storage could not be updated.");
  }
  announceChat("Conversation cleared.");
  renderMessages();
});

$("#refresh-advanced").addEventListener("click", () => void refresh());
$("#refresh-models").addEventListener("click", () => void loadLocalModels()
  .then(() => setModelActionState(false, "Local model scan complete."))
  .catch((error) => setModelActionState(false, error.message || "Local scan failed.")));
$("#model-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void searchModels($("#model-search-query").value);
});
$("#max-tokens").addEventListener("change", saveSettings);
$("#cancel-download").addEventListener("click", () => state.downloadController?.abort());

renderMessages();
void refresh();
