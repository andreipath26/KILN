const $ = (selector) => document.querySelector(selector);
const state = {
  model: null,
  messages: [],
  controller: null,
  busy: false,
  capabilities: null,
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
  return `kiln.ui.chat.v1:${state.model?.name || "unconfigured"}`;
}

function readHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(historyKey()) || "[]");
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
  container.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = '<div class="empty-glyph" aria-hidden="true">✳</div><strong>Your local chat starts here</strong><p>Send a message to begin. Your conversation is saved in this browser on this device.</p>';
    container.append(empty);
  } else {
    for (const message of state.messages) {
      const item = document.createElement("article");
      item.className = `message ${message.role}`;
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = message.role === "user" ? "You" : "KILN";
      const content = document.createElement("div");
      content.textContent = message.content;
      item.append(label, content);
      container.append(item);
    }
  }
  container.scrollTop = container.scrollHeight;
  const enabled = Boolean(state.model);
  $("#prompt").disabled = !enabled || state.busy;
  $("#send-button").disabled = !enabled || state.busy;
  $("#clear-chat").disabled = !enabled || !state.messages.length || state.busy;
  $("#chat-subtitle").textContent = state.model ? "Streaming from the local KILN API" : "Waiting for a loaded model";
}

function showChatError(message = "") {
  $("#chat-error").textContent = message;
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
    ? "This is the current CPU/RAM profile. Accelerator discovery and shared-memory topology are not yet exposed by the API."
    : "The runtime did not return a system profile.";
}

function renderSession(model, bootstrap) {
  const list = $("#session-details");
  list.replaceChildren();
  const rows = [
    ["Model", model?.name || "None loaded"],
    ["Model size", model ? `${(model.size / 1_073_741_824).toFixed(2)} GiB` : "—"],
    ["Session config", model ? "Loaded at API startup" : "Not configured"],
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
    const [versionResponse, tagsResponse, capabilitiesResponse] = await Promise.all([
      fetch("/api/version", { cache: "no-store" }),
      fetch("/api/tags", { cache: "no-store" }),
      fetch("/api/ui/capabilities", { cache: "no-store" }),
    ]);
    if (![versionResponse, tagsResponse, capabilitiesResponse].every((response) => response.ok)) {
      throw new Error("KILN API returned an error while loading the local runtime state.");
    }
    const [version, tags, capabilities] = await Promise.all([
      versionResponse.json(), tagsResponse.json(), capabilitiesResponse.json(),
    ]);
    state.model = tags.models?.[0] || null;
    setServerState(true, "Local runtime connected");
    renderSession(state.model, { api_version: version.version });
    renderCapabilities(capabilities);
    state.messages = readHistory();
    renderMessages();
  } catch (error) {
    state.model = null;
    setServerState(false, "Local runtime unavailable");
    renderSession(null, null);
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
  setBusy(true);
  state.controller = new AbortController();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: state.model.name, messages: requestMessages, stream: true }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.error || `Chat request failed (${response.status}).`);
    }
    if (!response.body) throw new Error("The browser does not support streaming this response.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop() || "";
      for (const line of lines) consumeChatLine(line, answer);
      renderMessages();
    }
    buffered += decoder.decode();
    if (buffered.trim()) consumeChatLine(buffered, answer);
    if (!answer.content && !state.controller.signal.aborted) {
      throw new Error("The model returned an empty response.");
    }
    saveHistory();
  } catch (error) {
    if (error.name === "AbortError") {
      if (answer.content) saveHistory();
      else state.messages.pop();
    } else {
      state.messages.pop();
      showChatError(error.message || "Inference failed.");
    }
  } finally {
    state.controller = null;
    setBusy(false);
    renderMessages();
    $("#prompt").focus();
  }
}

function consumeChatLine(line, answer) {
  if (!line.trim()) return;
  let event;
  try { event = JSON.parse(line); }
  catch { throw new Error("KILN returned an invalid streaming response."); }
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
  localStorage.removeItem(historyKey());
  showChatError();
  renderMessages();
});

$("#refresh-advanced").addEventListener("click", () => void refresh());

renderMessages();
void refresh();
