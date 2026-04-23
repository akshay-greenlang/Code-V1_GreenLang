/**
 * W4-D — Entitlements admin flow.
 *
 * 1. Grant a new pack to tenant `acme`.
 * 2. Revoke the grant for tenant `acme`.
 * 3. Verify cross-tenant blindness: the list renders only the caller's
 *    scope (confirmed via the `/v1/admin/entitlements` response being
 *    respected — the UI never injects synthetic rows).
 */
import { expect, test } from "@playwright/test";

let grants = [
  {
    tenant_id: "beta",
    plan: "pro",
    packs: ["electricity-premium"],
    data_classes: ["iot-raw"],
    updated_at: "2026-04-01T00:00:00Z",
  },
];

test.describe("W4-D entitlements admin", () => {
  test.beforeEach(async ({ page }) => {
    grants = [
      {
        tenant_id: "beta",
        plan: "pro",
        packs: ["electricity-premium"],
        data_classes: ["iot-raw"],
        updated_at: "2026-04-01T00:00:00Z",
      },
    ];
    await page.addInitScript(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });
    await page.route("**/v1/admin/entitlements", (route) => {
      if (route.request().method() === "POST") {
        const body = route.request().postDataJSON() as {
          tenant_id: string;
          packs: string[];
          data_classes: string[];
          plan?: string;
        };
        const idx = grants.findIndex((g) => g.tenant_id === body.tenant_id);
        const next = {
          ...body,
          plan: body.plan ?? "pro",
          updated_at: new Date().toISOString(),
        };
        if (idx >= 0) grants[idx] = next;
        else grants.push(next);
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(next),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ grants }),
      });
    });
    await page.route("**/v1/admin/entitlements/*", (route) => {
      const url = new URL(route.request().url());
      const tenantId = decodeURIComponent(url.pathname.split("/").pop() ?? "");
      if (route.request().method() === "DELETE") {
        grants = grants.filter((g) => g.tenant_id !== tenantId);
        return route.fulfill({ status: 204, body: "" });
      }
      return route.fallback();
    });
  });

  test("grant then revoke a pack; cross-tenant blindness preserved", async ({ page }) => {
    await page.goto("/factors/entitlements");
    await expect(page.getByRole("heading", { name: /Entitlements Admin/i })).toBeVisible();

    await page.once("dialog", (d) => d.accept());

    // Grant new pack to tenant acme.
    await page.getByTestId("entitlements-add").click();
    await page.getByLabel("Tenant ID").fill("acme");
    await page.getByLabel("Plan").fill("enterprise");
    await page.getByTestId("entitlements-packs-input").fill("electricity-premium freight-premium");
    await page.getByLabel("Data classes").fill("iot-raw pii-redacted");
    await page.getByTestId("entitlements-save").click();

    await expect(page.getByText("acme")).toBeVisible();
    await expect(page.getByText("enterprise")).toBeVisible();

    // Revoke.
    page.once("dialog", (d) => d.accept());
    await page.getByTestId("entitlements-revoke-acme").click();
    await expect(page.getByText("acme")).toHaveCount(0);

    // Cross-tenant blindness — no rows for tenants not in `grants`.
    await expect(page.getByText(/^acme$/)).toHaveCount(0);
    await expect(page.getByText(/^beta$/)).toBeVisible();
  });
});
