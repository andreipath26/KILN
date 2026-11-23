const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const AxeBuilder = require("@axe-core/playwright").default;

const baseUrl = process.env.KILN_UI_URL || "http://127.0.0.1:11435";

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function localModels(activePath = "/tmp/test-model.gguf") {
  return [
    {
      name: "test-model.gguf",
      path: "/tmp/test-model.gguf",
      size_bytes: 1_000_000_000,
      source: "local GGUF",
      valid_gguf: true,
      active: activePath === "/tmp/test-model.gguf",
    },
    {
      name: "other-model.gguf",
      path: "/tmp/other-model.gguf",
      size_bytes: 500_000_000,
      source: "local GGUF",
      native_stream_candidate: true,
      valid_gguf: true,
      active: activePath === "/tmp/other-model.gguf",
    },
  ];
}

function conversation(id, updatedAt = "2026-11-23T10:00:00.000Z") {
  return {
    id,
    title: `Conversation ${id}`,
    updated_at: updatedAt,
    messages: [{ role: "user", content: `Prompt ${id}` }],
  };
}

function conversationBackup(conversations = [], legacyChats = []) {
  return {
    format: "kiln-conversations-v2",
    exported_at: "2026-11-23T10:00:00.000Z",
    conversations,
    legacy_chats: legacyChats,
  };
}

async function installApiMocks(page, { modelLoaded = true, autoStartStreamFallback = false } = {}) {
  let activePath = modelLoaded ? "/tmp/test-model.gguf" : null;
  const activationAttempts = [];
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    if (url.pathname === "/api/version") {
      return json(route, { version: "browser-test" });
    }
    if (url.pathname === "/api/tags") {
      return json(route, {
        models: activePath ? [{ name: "test-model", size: 1_000_000_000 }] : [],
      });
    }
    if (url.pathname === "/api/ui/capabilities") {
      return json(route, {
        schema_version: 1,
        api_version: "browser-test",
        active_model: activePath ? { name: "test-model" } : null,
        system_profile: {
          tier: "T0",
          cpu_cores: 4,
          ram_gb: 16,
          n_ctx: 2048,
          n_batch: 512,
          n_threads: 4,
        },
        capabilities: [{
          id: "inference.chat_stream",
          label: "Streaming chat",
          status: "available",
          detail: "Mocked response for browser verification.",
        }],
      });
    }
    if (url.pathname === "/api/ui/models" && request.method() === "GET") {
      const startupCandidates = autoStartStreamFallback && !activePath ? [
        { model_path: "/tmp/test-model.gguf", model_name: "resident model", mode: "standard" },
        { model_path: "/tmp/other-model.gguf", model_name: "streamable MoE", mode: "experimental_cpu_moe_stream" },
      ] : [];
      return json(route, {
        models: activePath || autoStartStreamFallback ? localModels(activePath) : [],
        recommendation: startupCandidates[0] || null,
        startup_candidates: startupCandidates,
      });
    }
    if (url.pathname === "/api/ui/models/search") {
      const query = url.searchParams.get("q");
      if (query === "incomplete split") {
        return json(route, {models:[{
          repository:"kiln-test/incomplete-split",downloads:1,license:"MIT",gated:false,
          artifacts:[
            {path:"flash-00001-of-00003.gguf",size_bytes:1024},
            {path:"flash-00003-of-00003.gguf",size_bytes:1024},
          ],
        }]});
      }
      if (query === "complete split") {
        return json(route, {models:[{
          repository:"kiln-test/complete-split",downloads:1,license:"MIT",gated:false,
          artifacts:[
            {path:"flash-00001-of-00002.gguf",size_bytes:1024},
            {path:"flash-00002-of-00002.gguf",size_bytes:1024},
          ],
        }]});
      }
      return json(route, {
        models: [{
          repository: "kiln-test/tiny-model",
          downloads: 123,
          license: "Apache-2.0",
          gated: false,
          artifacts: [{ path: "tiny.Q4_K_M.gguf", size_bytes: 1_048_576 }],
        }],
      });
    }
    if (url.pathname === "/api/ui/models/pull") {
      return route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: [
          JSON.stringify({ event: "progress", bytes_received: 524288, total_bytes: 1048576 }),
          JSON.stringify({ event: "done", path: "/tmp/tiny.Q4_K_M.gguf" }),
          "",
        ].join("\n"),
      });
    }
    if (url.pathname === "/api/ui/models/activate") {
      const body = request.postDataJSON();
      activationAttempts.push({ path: body.path, mode: body.mode });
      if (autoStartStreamFallback && body.path === "/tmp/test-model.gguf") {
        return json(route, { error: "mock resident model backend unsupported" }, 422);
      }
      if (body.path === "/tmp/other-model.gguf"
          && !(autoStartStreamFallback && body.mode === "stream")) {
        return json(route, { error: "mock model load failed; previous model restored" }, 422);
      }
      activePath = body.path;
      return json(route, { model: "test-model", active: true });
    }
    if (url.pathname === "/api/chat") {
      return route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: [
          JSON.stringify({ model: "test-model", message: { role: "assistant", content: "Browser " }, done: false }),
          JSON.stringify({ model: "test-model", message: { role: "assistant", content: "test response." }, done: false }),
          JSON.stringify({ model: "test-model", message: { role: "assistant", content: "" }, done: true }),
          "",
        ].join("\n"),
      });
    }
    return json(route, { error: `unhandled API request: ${url.pathname}` }, 404);
  });
  return activationAttempts;
}

async function checkAccessibility(page, label, { advanced = false } = {}) {
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter((item) => ["critical", "serious"].includes(item.impact));
  assert.deepEqual(
    serious.map(({ id, impact, description, nodes }) => ({
      id,
      impact,
      description,
      targets: nodes.map((node) => node.target),
    })),
    [],
    `${label} has serious or critical accessibility violations`,
  );
  const accessibilityTree = await page.locator("body").ariaSnapshot();
  assert.match(accessibilityTree, /tablist "KILN navigation"/);
  assert.match(accessibilityTree, advanced ? /tab "System details" \[selected\]/ : /tab "Chat"/);
  assert.match(accessibilityTree, advanced ? /spinbutton "Chat output limit"/ : /textbox "Message your model"/);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const failures = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();
    page.on("pageerror", (error) => failures.push(error.message));
    await installApiMocks(page);
    const response = await page.goto(baseUrl, { waitUntil: "networkidle" });
    assert.equal(response.status(), 200, "the KILN API serves the bundled UI");
    const headers = response.headers();
    assert.match(headers["content-security-policy"], /script-src 'self'/);
    assert.equal(headers["x-content-type-options"], "nosniff");

    await page.getByRole("heading", { name: "What can we work on?" }).waitFor();
    await page.waitForFunction(() => document.querySelector("#active-model-name")?.textContent === "test-model");
    await checkAccessibility(page, "Simple view");

    await page.locator("#prompt").fill("Say hello");
    await page.getByRole("button", { name: "Send" }).click();
    await page.getByText("Browser test response.", { exact: true }).waitFor();
    assert.equal(await page.locator(".message.user").count(), 1);

    await page.getByRole("tab", { name: "Models" }).click();
    // Only the verified eligible MoE fixture offers the experimental action.
    assert.equal(await page.getByRole("button", {name:"Try streaming"}).count(),1);
    await page.locator("#model-search-query").fill("tiny model");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await page.getByText("kiln-test/tiny-model", { exact: true }).waitFor();
    await page.getByRole("button", { name: /Download tiny.Q4_K_M.gguf/ }).click();
    await page.getByText("Verified and installed tiny.Q4_K_M.gguf.", { exact: false }).waitFor();

    // A repository with a missing split part must never display a button
    // that silently downloads a single incomplete shard.
    await page.locator("#model-search-query").fill("incomplete split");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    const incomplete = page.locator("#remote-models");
    await incomplete.getByText("kiln-test/incomplete-split", {exact:true}).waitFor();
    const blocked = incomplete.getByRole("button", {name:"Incomplete split model (download unavailable)"});
    assert.equal(await blocked.count(),1);
    assert.equal(await blocked.isDisabled(),true);
    assert.match(await blocked.getAttribute("title"),/not listed/);
    assert.equal(await incomplete.getByRole("button", {name:/Download flash-00003/}).count(),0);

    await page.locator("#model-search-query").fill("complete split");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    const complete = page.locator("#remote-models");
    await complete.getByText("kiln-test/complete-split", {exact:true}).waitFor();
    assert.equal(await complete.getByRole("button",{name:"Download complete model (2 shards)"}).count(),1);
    assert.equal(await complete.getByRole("button",{name:/Download flash-00002/}).count(),0);

    await page.getByRole("button", { name: "Use model" }).click();
    await page.locator("#auto-setup-status")
      .getByText("mock model load failed; previous model restored", { exact: false }).waitFor();
    await page.getByText("test-model.gguf", { exact: true }).waitFor();

    const simpleTab = page.getByRole("tab", { name: "Chat" });
    await simpleTab.focus();
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");
    const advancedTab = page.getByRole("tab", { name: "System details" });
    await advancedTab.waitFor({ state: "visible" });
    assert.equal(await advancedTab.getAttribute("aria-selected"), "true");
    await page.getByRole("heading", { name: "Available in this build" }).waitFor();
    await checkAccessibility(page, "Advanced view", { advanced: true });
    await page.locator("#max-tokens").fill("512");
    await page.locator("#max-tokens").dispatchEvent("change");
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("kiln.ui.settings.v1:/tmp/test-model.gguf")).max_tokens), 512);
    await advancedTab.focus();
    await page.keyboard.press("Home");
    assert.equal(await simpleTab.getAttribute("aria-selected"), "true");
    await simpleTab.focus();
    await page.keyboard.press("End");
    assert.equal(await advancedTab.getAttribute("aria-selected"), "true");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: "networkidle" });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, "mobile layout has no horizontal overflow");
    await checkAccessibility(page, "narrow Simple view");
    await page.emulateMedia({ forcedColors: "active" });
    await page.locator("#prompt").focus();
    assert.equal(await page.locator("#prompt").evaluate((element) => document.activeElement === element), true, "keyboard focus remains visible and reachable in forced-colors mode");
    assert.equal(await page.locator(".composer").evaluate((element) => {
      const style = getComputedStyle(element);
      return style.outlineStyle !== "none" && Number.parseFloat(style.outlineWidth) >= 2;
    }), true, "the chat composer has a visible system-color focus ring in forced-colors mode");
    await context.close();

    const emptyContext = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const emptyPage = await emptyContext.newPage();
    const activationAttempts = await installApiMocks(emptyPage, {
      modelLoaded: false,
      autoStartStreamFallback: true,
    });
    await emptyPage.goto(baseUrl, { waitUntil: "networkidle" });
    const setupStatus = emptyPage.locator("#machine-setup-status");
    await setupStatus.waitFor({ state: "visible" });
    assert.equal(
      (await setupStatus.textContent()).trim(),
      "Ready. KILN assessed this machine and started test-model with experimental CPU expert streaming.",
      "automatic setup outcome is visible from the default Chat destination",
    );
    assert.deepEqual(activationAttempts, [
      { path: "/tmp/test-model.gguf", mode: "auto" },
      { path: "/tmp/other-model.gguf", mode: "stream" },
    ], "dashboard startup falls back from a rejected resident model to the recommended streaming mode");
    assert.equal(await emptyPage.locator("#send-button").isDisabled(), false);
    await checkAccessibility(emptyPage, "automatically configured model");
    await emptyContext.close();

    const backupContext = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const backupPage = await backupContext.newPage();
    backupPage.on("pageerror", (error) => failures.push(error.message));
    await installApiMocks(backupPage);
    await backupPage.goto(baseUrl, { waitUntil: "networkidle" });
    await backupPage.getByRole("tab", { name: "Activity" }).click();
    backupPage.on("dialog", (dialog) => dialog.accept());

    const legacyBackup = conversationBackup([], [{
      model_key: "test-model.gguf",
      messages: [{ role: "user", content: "Legacy prompt" }, { role: "assistant", content: "Legacy answer" }],
    }]);
    const legacyFile = {
      name: "legacy-backup.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(legacyBackup)),
    };
    const importInput = backupPage.locator("#import-conversations-file");
    const importStatus = backupPage.locator("#conversation-data-status");
    const modelHistoryKey = "kiln.ui.conversations.v1:/tmp/test-model.gguf";
    const initialModelHistory = await backupPage.evaluate((key) =>
      JSON.parse(localStorage.getItem(key)), modelHistoryKey);
    await importInput.setInputFiles(legacyFile);
    await importStatus.getByText("Imported and merged 1 conversations.", { exact: false }).waitFor();
    const importedLegacy = await backupPage.evaluate(() =>
      JSON.parse(localStorage.getItem("kiln.ui.conversations.v1:/tmp/test-model.gguf")));
    assert.equal(importedLegacy.length, initialModelHistory.length + 1);
    const legacyId = importedLegacy.find((item) => item.id.startsWith("legacy-"))?.id;
    assert.match(legacyId, /^legacy-[0-9a-f]{16}$/);
    await importStatus.evaluate((element) => { element.textContent = ""; });
    await importInput.setInputFiles(legacyFile);
    await importStatus.getByText("Imported and merged 1 conversations.", { exact: false }).waitFor();
    const reimportedLegacy = await backupPage.evaluate(() =>
      JSON.parse(localStorage.getItem("kiln.ui.conversations.v1:/tmp/test-model.gguf")));
    assert.equal(reimportedLegacy.length, initialModelHistory.length + 1,
      "reimporting the same legacy backup does not duplicate its conversation");
    assert.equal(reimportedLegacy.filter((item) => item.id.startsWith("legacy-")).length, 1);
    assert.equal(reimportedLegacy.find((item) => item.id.startsWith("legacy-"))?.id, legacyId);

    const otherModelKey = "kiln.ui.conversations.v1:/tmp/other-model.gguf";
    const fullHistory = Array.from({ length: 50 }, (_, index) => conversation(`existing-${index}`));
    await backupPage.evaluate(({ key, items }) => localStorage.setItem(key, JSON.stringify(items)), {
      key: otherModelKey,
      items: fullHistory,
    });
    const goodModelKey = modelHistoryKey;
    const beforeGoodModel = await backupPage.evaluate((key) => localStorage.getItem(key), goodModelKey);
    const overflowingBackup = conversationBackup([
      { model_key: "/tmp/test-model.gguf", conversations: [conversation("valid-new-import")] },
      { model_key: "/tmp/other-model.gguf", conversations: [conversation("overflow-new-import")] },
    ]);
    await importInput.setInputFiles({
      name: "overflowing-backup.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(overflowingBackup)),
    });
    await importStatus.getByText(/would exceed the 50-conversation limit/).waitFor();
    assert.equal(
      await backupPage.evaluate((key) => localStorage.getItem(key), goodModelKey),
      beforeGoodModel,
      "an overflow in a later model group leaves earlier model histories unchanged",
    );
    assert.deepEqual(
      await backupPage.evaluate((key) => JSON.parse(localStorage.getItem(key)), otherModelKey),
      fullHistory,
      "the overflowing model history remains unchanged",
    );
    await backupContext.close();

    assert.deepEqual(failures, [], "the UI has no uncaught JavaScript errors");
    console.log("KILN UI browser checks passed: chat, model search/download, activation recovery, automatic setup, keyboard modes, accessibility, mobile layout, and backup import safety.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
