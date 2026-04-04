import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { platform } from "node:os";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));
const snapDir = join(__dirname, "visual-shell.spec.ts-snapshots");
const isCi = process.env.CI === "true";

function dirHasPngRecursive(dir: string): boolean {
  if (!existsSync(dir)) return false;
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (dirHasPngRecursive(full)) return true;
    } else if (name.endsWith(".png")) {
      return true;
    }
  }
  return false;
}

const hasPngBaselines = dirHasPngRecursive(snapDir);

test.describe("Visual regression (committed baselines)", () => {
  // eslint-disable-next-line no-empty-pattern -- @playwright/test requires object-destructured first argument
  test.beforeEach(({}, testInfo) => {
    if (isCi && !hasPngBaselines && testInfo.config.updateSnapshots === "none") {
      testInfo.skip(
        true,
        "No PNG baselines in e2e/visual-shell.spec.ts-snapshots/. Generate and commit: npx playwright test e2e/visual-shell.spec.ts --update-snapshots (PLAYWRIGHT_PREBUILT=1 recommended), or run the v2-shell-visual-snapshots workflow."
      );
    }
  });

  test("CBAM workspace viewport", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await expect(page).toHaveScreenshot("cbam-workspace.png", {
      fullPage: false,
      animations: "disabled"
    });
  });

  test("Run center viewport", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/runs");
    await page.getByRole("heading", { name: /Run Center/i }).waitFor();
    await expect(page).toHaveScreenshot("runs-center.png", { fullPage: false, animations: "disabled" });
  });

  test("Governance viewport (auditor)", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/apps/cbam");
    await page.evaluate(() => window.localStorage.setItem("gl.v2.shell.role", "auditor"));
    await page.goto("/governance");
    await page.getByRole("heading", { name: /Governance Center/i }).waitFor();
    await expect(page).toHaveScreenshot("governance.png", { fullPage: false, animations: "disabled" });
  });

  test("Admin viewport", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/apps/cbam");
    await page.evaluate(() => window.localStorage.setItem("gl.v2.shell.role", "admin"));
    await page.goto("/admin");
    await page.getByRole("heading", { name: /Admin Console/i }).waitFor();
    await expect(page).toHaveScreenshot("admin-console.png", { fullPage: false, animations: "disabled" });
  });

  test("Command palette open", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    const isMac = platform() === "darwin";
    await page.keyboard.press(isMac ? "Meta+k" : "Control+k");
    await page.getByRole("heading", { name: /Command Palette/i }).waitFor();
    await expect(page).toHaveScreenshot("command-palette.png", { fullPage: false, animations: "disabled" });
  });
});
