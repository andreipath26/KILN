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
      valid_gguf: true,
      active: activePath === "/tmp/other-model.gguf",
    },
  ];
}

async function installApiMocks(page, { modelLoaded = true } = {}) {
  let activePath = modelLoaded ? "/tmp/test-model.gguf" : null;
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
      return json(route, { models: activePath ? localModels(activePath) : [] });
    }
    if (url.pathname === "/api/ui/models/search") {
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
      if (body.path === "/tmp/other-model.gguf") {
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
  assert.match(accessibilityTree, /tablist "KILN modes"/);
  assert.match(accessibilityTree, advanced ? /tab "Advanced" \[selected\]/ : /tab "Simple"/);
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

    await page.getByRole("heading", { name: "Talk to your model" }).waitFor();
    await page.getByText("test-model.gguf", { exact: true }).waitFor();
    await checkAccessibility(page, "Simple view");

    await page.locator("#prompt").fill("Say hello");
    await page.getByRole("button", { name: "Send" }).click();
    await page.getByText("Browser test response.", { exact: true }).waitFor();
    assert.equal(await page.locator(".message.user").count(), 1);

    await page.locator("#model-search-query").fill("tiny model");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await page.getByText("kiln-test/tiny-model", { exact: true }).waitFor();
    await page.getByRole("button", { name: /Download tiny.Q4_K_M.gguf/ }).click();
    await page.getByText("Verified and installed tiny.Q4_K_M.gguf.", { exact: false }).waitFor();

    await page.getByRole("button", { name: "Use model" }).click();
    await page.getByText("mock model load failed; previous model restored", { exact: true }).waitFor();
    await page.getByText("test-model.gguf", { exact: true }).waitFor();

    const simpleTab = page.getByRole("tab", { name: "Simple" });
    await simpleTab.focus();
    await page.keyboard.press("ArrowRight");
    const advancedTab = page.getByRole("tab", { name: "Advanced" });
    await advancedTab.waitFor({ state: "visible" });
    assert.equal(await advancedTab.getAttribute("aria-selected"), "true");
    await page.getByRole("heading", { name: "Available in this build" }).waitFor();
    await checkAccessibility(page, "Advanced view", { advanced: true });
    await page.locator("#max-tokens").fill("512");
    await page.locator("#max-tokens").dispatchEvent("change");
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("kiln.ui.settings.v1:/tmp/test-model.gguf")).max_tokens), 512);
    await advancedTab.focus();
    await page.keyboard.press("ArrowLeft");
    assert.equal(await simpleTab.getAttribute("aria-selected"), "true");

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
    await installApiMocks(emptyPage, { modelLoaded: false });
    await emptyPage.goto(baseUrl, { waitUntil: "networkidle" });
    await emptyPage.getByText("No model is loaded in the API server.", { exact: false }).waitFor();
    assert.equal(await emptyPage.locator("#send-button").isDisabled(), true);
    await checkAccessibility(emptyPage, "no-model state");
    await emptyContext.close();

    assert.deepEqual(failures, [], "the UI has no uncaught JavaScript errors");
    console.log("KILN UI browser checks passed: chat, model search/download, activation recovery, keyboard modes, accessibility, mobile layout, and no-model startup.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
