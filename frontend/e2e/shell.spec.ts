import { expect, test } from "@playwright/test";
import { assertNoSeriousAxeViolations } from "./axe-helpers";

test.describe("V2.2 shell", () => {
  test("CBAM workspace heading renders", async ({ page }) => {
    await page.goto("/apps/cbam");
    await expect(page.getByRole("heading", { name: /CBAM Workspace/i })).toBeVisible();
  });

  test("admin console visible for admin role", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });
    await page.goto("/admin");
    await expect(page.getByRole("heading", { name: /Admin Console/i })).toBeVisible();
  });

  test("run center loads", async ({ page }) => {
    await page.goto("/runs");
    await expect(page.getByRole("heading", { name: /Run Center/i })).toBeVisible();
  });

  test("governance center loads for auditor role", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "auditor");
    });
    await page.goto("/governance");
    await expect(page.getByRole("heading", { name: /Governance Center/i })).toBeVisible();
  });

  test("CBAM workspace has no serious axe violations (incl. color-contrast)", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.getByRole("heading", { name: /CBAM Workspace/i }).waitFor();
    await assertNoSeriousAxeViolations(page);
  });

  test("runs page has no serious axe violations", async ({ page }) => {
    await page.goto("/runs");
    await page.getByRole("heading", { name: /Run Center/i }).waitFor();
    await assertNoSeriousAxeViolations(page);
  });

  test("governance page has no serious axe violations", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "auditor");
    });
    await page.goto("/governance");
    await page.getByRole("heading", { name: /Governance Center/i }).waitFor();
    await assertNoSeriousAxeViolations(page);
  });

  test("admin page has no serious axe violations", async ({ page }) => {
    await page.goto("/apps/cbam");
    await page.evaluate(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });
    await page.goto("/admin");
    await page.getByRole("heading", { name: /Admin Console/i }).waitFor();
    await assertNoSeriousAxeViolations(page);
  });
});
