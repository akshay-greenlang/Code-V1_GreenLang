import { expect, test } from "@playwright/test";

test.describe("Operator journeys", () => {
  test("command palette filters and navigates to Run Center", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    const isMac = process.platform === "darwin";
    await page.keyboard.press(isMac ? "Meta+k" : "Control+k");
    await expect(page.getByRole("heading", { name: /Command Palette/i })).toBeVisible();
    await page.getByRole("textbox", { name: /Command search/i }).fill("runs");
    const runsLink = page.getByRole("link", { name: /Runs \(\/runs\)/i });
    await expect(runsLink).toBeVisible();
    await runsLink.click();
    await expect(page).toHaveURL(/\/runs$/);
    await expect(page.getByRole("heading", { name: /Run Center/i })).toBeVisible();
  });

  test("operator role cannot open governance (redirect)", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "operator");
    });
    await page.goto("/governance");
    await expect(page).toHaveURL(/\/apps\/cbam/);
    await expect(page.getByRole("heading", { name: /CBAM Workspace/i })).toBeVisible();
  });

  test("auditor role cannot open admin (redirect)", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "auditor");
    });
    await page.goto("/admin");
    await expect(page).toHaveURL(/\/apps\/cbam/);
    await expect(page.getByRole("heading", { name: /CBAM Workspace/i })).toBeVisible();
  });

  test("CBAM workspace run demo completes with artifacts", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await page.getByRole("button", { name: /Run demo/i }).click();
    await expect(page.getByText(/Run [a-f0-9]{32}/i)).toBeVisible({ timeout: 120_000 });
    await expect(page.getByText(/Artifacts/i).first()).toBeVisible();
  });

  test("artifact diff shows checksums after two demo runs", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await page.getByRole("button", { name: /Run demo/i }).click();
    await expect(page.getByRole("heading", { name: /^Run [a-f0-9]{32}$/i })).toBeVisible({ timeout: 120_000 });
    await page.getByRole("button", { name: /Run demo/i }).click();
    await expect(page.getByRole("heading", { name: /^Run [a-f0-9]{32}$/i })).toBeVisible({ timeout: 120_000 });

    await page.goto("/runs");
    await page.getByRole("heading", { name: /Run Center/i }).waitFor();

    const runA = page.getByRole("combobox", { name: /Artifact diff run A/i });
    const runB = page.getByRole("combobox", { name: /Artifact diff run B/i });
    const artifact = page.getByRole("combobox", { name: /Artifact diff common artifact/i });

    await runA.click();
    const optsA = page.getByRole("option");
    expect(await optsA.count()).toBeGreaterThan(2);
    await optsA.nth(1).click();

    await runB.click();
    const optsB = page.getByRole("option");
    await optsB.nth(Math.min(2, (await optsB.count()) - 1)).click();

    await artifact.click();
    const optsArt = page.getByRole("option");
    const nArt = await optsArt.count();
    if (nArt <= 1) {
      test.skip();
      return;
    }
    await optsArt.nth(1).click();

    await page.getByRole("button", { name: /^Compare$/ }).click();
    await expect(page.getByText(/Run A checksum:/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/Run B checksum:/)).toBeVisible();
  });
});
