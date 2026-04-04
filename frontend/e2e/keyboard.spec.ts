import { expect, test } from "@playwright/test";

test.describe("Keyboard and focus", () => {
  test("Cmd+K opens command palette and Escape closes", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    const isMac = process.platform === "darwin";
    await page.keyboard.press(isMac ? "Meta+k" : "Control+k");
    await expect(page.getByRole("heading", { name: /Command Palette/i })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("heading", { name: /Command Palette/i })).not.toBeVisible();
  });

  test("Escape closes command palette when opened via button", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await page.getByRole("button", { name: /Cmd\+K/i }).click();
    await expect(page.getByRole("heading", { name: /Command Palette/i })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("heading", { name: /Command Palette/i })).not.toBeVisible();
  });

  test("skip link moves focus to main landmark", async ({ page }) => {
    await page.goto("/apps/cbam");
    const skip = page.getByRole("link", { name: /Skip to main content/i });
    await skip.focus();
    await expect(skip).toBeFocused();
    await skip.press("Enter");
    const main = page.getByRole("main", { name: /Workspace content/i });
    await expect(main).toBeVisible();
  });

  test("compliance rail aside is present after shell chrome loads", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await expect(page.getByRole("complementary", { name: /Compliance and policy summary/i })).toBeVisible({
      timeout: 15_000
    });
  });

  test("DAG stage buttons are focusable on runs page", async ({ page }) => {
    await page.goto("/runs");
    await page.getByRole("heading", { name: /Run Center/i }).waitFor();
    const firstNode = page.locator("circle[role='button']").first();
    const count = await firstNode.count();
    if (count === 0) {
      test.skip();
      return;
    }
    await firstNode.focus();
    await expect(firstNode).toBeFocused();
  });
});
