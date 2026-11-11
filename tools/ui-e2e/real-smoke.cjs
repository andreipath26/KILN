const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.env.KILN_UI_URL || "http://127.0.0.1:11435";

async function main() {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const response = await page.goto(baseUrl, { waitUntil: "networkidle" });
    assert.equal(response.status(), 200);
    await page.getByRole("heading", { name: "Talk to your model" }).waitFor();
    // Test the real module-scoped UI renderer using a synthetic repository
    // response: no network download or exposed test-only global is needed.
    const fakeSearch = "**/api/ui/models/search?*";
    await page.route(fakeSearch, async (route) => {
      await route.fulfill({ status: 200, contentType: "application/json",
        body: JSON.stringify({ models: [{
          repository: "fixture/split", license: "mit", gated: false, downloads: 1,
          artifacts: [
            { path: "quant/model-Q2_K-00001-of-00002.gguf", size_bytes: 256 },
            { path: "quant/model-Q2_K-00002-of-00002.gguf", size_bytes: 256 },
            { path: "quant/model-Q4_K-00001-of-00002.gguf", size_bytes: 256 },
          ],
        }] }),
      });
    });
    await page.locator("#model-search-query").fill("fixture");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await page.locator("#remote-models").getByRole("button", {
      name: "Download complete model (2 shards)",
    }).waitFor();
    const artifactButtons = await page.locator("#remote-models button").allTextContents();
    assert.equal(artifactButtons.filter((text) => text.startsWith("Download complete model")).length, 1);
    assert.equal(artifactButtons.length, 2, "partial quantization remains a single-file option");
    await page.unroute(fakeSearch);
    await page.getByText("SmolLM2-135M-Instruct.Q4_K_M.gguf", { exact: true }).waitFor();
    assert.equal(await page.locator("#send-button").isDisabled(), false);

    const alternate = page.locator(".model-entry").filter({ hasText: "alternate-smoke.gguf" });
    await alternate.getByRole("button", { name: "Use model" }).click();
    await page.getByText("alternate-smoke", { exact: true }).first().waitFor();
    await page.waitForFunction(() => document.querySelector("#active-model-name").textContent === "alternate-smoke");

    const invalid = page.locator(".model-entry").filter({ hasText: "zz-invalid-arch.gguf" });
    await invalid.getByRole("button", { name: "Use model" }).click();
    await page.getByText(/restored previous model 'alternate-smoke'/).waitFor();
    assert.equal(await page.locator("#active-model-name").textContent(), "alternate-smoke");

    await page.getByRole("tab", { name: "Advanced" }).click();
    const profileText = await page.locator("#system-profile").textContent();
    assert.ok(profileText.includes("Available RAM"), "UI exposes current memory headroom");
    assert.ok(profileText.includes("Estimated model residency budget"), "UI exposes conservative model-loading limit");
    assert.match(await page.locator("#session-details").textContent(),
      /Standard \/ mmap model loading/, "UI shows the actual current engine mode");
    await page.locator("#max-tokens").fill("16");
    await page.locator("#max-tokens").dispatchEvent("change");
    await page.getByRole("tab", { name: "Simple" }).click();
    await page.locator("#prompt").fill("Reply with one short greeting.");
    await page.getByRole("button", { name: "Send" }).click();
    await page.waitForFunction(() => {
      const answer = document.querySelector(".message.assistant div");
      return Boolean(answer?.textContent.trim());
    }, { timeout: 60_000 });

    const answer = (await page.locator(".message.assistant div").textContent()).trim();
    assert.ok(answer.length > 0, "the live model returned assistant text");
    assert.deepEqual(pageErrors, [], "real-model chat produced no uncaught UI errors");

    const modelsPage = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    modelsPage.on("pageerror", (error) => pageErrors.push(error.message));
    await modelsPage.goto(baseUrl, { waitUntil: "networkidle" });
    await modelsPage.locator("#model-search-query").fill("QuantFactory/SmolLM2-135M-Instruct-GGUF");
    await modelsPage.getByRole("button", { name: "Search", exact: true }).click();
    await modelsPage.waitForFunction(() => /repositories found\./.test(document.querySelector("#model-library-status").textContent), { timeout: 30_000 });
    const remoteResults = modelsPage.locator("#remote-models");
    const resultTitle = remoteResults.getByText("QuantFactory/SmolLM2-135M-Instruct-GGUF", { exact: true });
    await resultTitle.waitFor();
    await remoteResults.getByRole("button", { name: /Download SmolLM2-135M-Instruct.Q2_K.gguf/ }).click();
    await modelsPage.getByText(/Downloading SmolLM2-135M-Instruct.Q2_K.gguf/).waitFor();
    await modelsPage.getByRole("button", { name: "Cancel download" }).click();
    await modelsPage.getByText("Download canceled; the partial file was removed.", { exact: true }).waitFor();

    const hasPartialDownload = () => {
      const visit = (directory) => fs.readdirSync(directory, { withFileTypes: true }).some((entry) => {
        const entryPath = path.join(directory, entry.name);
        return entry.isDirectory() ? visit(entryPath) : entry.name.endsWith(".part");
      });
      return visit(process.env.KILN_MODELS_DIR);
    };
    const cleanupDeadline = Date.now() + 60_000;
    while (hasPartialDownload() && Date.now() < cleanupDeadline) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    assert.equal(hasPartialDownload(), false, "canceling a real model download removes its partial file");
    console.log(`KILN real-model browser smoke passed: activation, rollback, chat (${answer.length} response characters), and canceled-download cleanup.`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
