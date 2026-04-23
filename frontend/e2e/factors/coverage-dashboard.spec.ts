/**
 * W4-D — Public coverage dashboard.
 *
 * Loads without auth. Asserts the Certified / Preview / Connector-only
 * totals and the per-family × per-jurisdiction matrix.
 */
import { expect, test } from "@playwright/test";

const COVERAGE_RESPONSE = {
  edition_id: "2026.Q1-electricity",
  generated_at: "2026-04-22T10:00:00Z",
  totals: {
    certified: 1240,
    preview: 312,
    connector_only: 88,
    all: 1640,
  },
  by_family: [
    { family: "electricity", certified: 420, preview: 44, connector_only: 11, all: 475 },
    { family: "fuel", certified: 230, preview: 17, connector_only: 3, all: 250 },
  ],
  by_family_jurisdiction: [
    {
      family: "electricity",
      jurisdiction: "IN",
      certified: 84,
      preview: 8,
      connector_only: 2,
      all: 94,
    },
    {
      family: "electricity",
      jurisdiction: "US",
      certified: 112,
      preview: 14,
      connector_only: 4,
      all: 130,
    },
    {
      family: "fuel",
      jurisdiction: "UK",
      certified: 71,
      preview: 5,
      connector_only: 1,
      all: 77,
    },
  ],
};

test.describe("W4-D public coverage dashboard", () => {
  test.beforeEach(async ({ page }) => {
    // Explicitly clear any dev role to simulate an anonymous visitor.
    await page.addInitScript(() => {
      window.localStorage.removeItem("gl.v2.shell.role");
      window.localStorage.removeItem("gl.auth.token");
    });
    await page.route("**/v1/coverage", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(COVERAGE_RESPONSE),
      }),
    );
  });

  test("loads without auth and shows expected counts + matrix", async ({ page }) => {
    await page.goto("/factors/coverage");
    await expect(page.getByRole("heading", { name: /Public Coverage/i })).toBeVisible();

    await expect(page.getByTestId("coverage-totals-certified")).toContainText("1,240");
    await expect(page.getByTestId("coverage-totals-preview")).toContainText("312");
    await expect(page.getByTestId("coverage-totals-connector-only")).toContainText("88");

    // Family × jurisdiction rows.
    await expect(page.getByText("electricity").first()).toBeVisible();
    await expect(page.getByText(/^IN$/)).toBeVisible();
    await expect(page.getByText(/^US$/)).toBeVisible();
    await expect(page.getByText(/^UK$/)).toBeVisible();

    // Confirmed no auth redirect.
    await expect(page).toHaveURL(/\/factors\/coverage$/);
  });
});
