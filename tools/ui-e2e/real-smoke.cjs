const assert = require("node:assert/strict");
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
    await page.getByText("SmolLM2-135M-Instruct.Q4_K_M.gguf", { exact: true }).waitFor();
    assert.equal(await page.locator("#send-button").isDisabled(), false);

    await page.getByRole("tab", { name: "Advanced" }).click();
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
    console.log(`KILN real-model browser smoke test passed (${answer.length} response characters).`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
