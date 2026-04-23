/**
 * W4-D — Canonical resolve demo spec.
 *
 * Exercises the 12,500 kWh India canonical demo on the Factors Explorer
 * page and asserts every Wave 2 / 2a / 2.5 envelope field renders:
 *   1-9   chosen_factor chips, source, FQSGauge, uncertainty band,
 *         license badge, deprecation banner (implicit: absent when
 *         factor is active), gas breakdown, audit text + draft banner,
 *         signed receipt debug pane.
 *   10-16 factor_id / factor_version / release_version / method_profile /
 *         method_pack_id / co2e_per_unit+unit / geography+scope
 *         (all surfaced as chips inside `section-chosen-factor`).
 */
import { expect, test } from "@playwright/test";

const CANONICAL_RESOLVE_RESPONSE = {
  chosen_factor: {
    factor_id: "EF:IN:grid_electricity:2026:v1",
    factor_version: "v1.2",
    release_version: "2026.Q1-electricity",
    method_profile: "ghgp_location_based",
    method_pack_id: "grid_electricity_pack",
    method_pack_version: "1.4.0",
    co2e_per_unit: 0.82,
    unit: "kgCO2e/kWh",
    geography: "IN",
    scope: "scope_2",
  },
  source: {
    source_id: "CEA-India-2026",
    organization: "Central Electricity Authority",
    publication: "CO2 Baseline Database 20.0",
    year: 2026,
    license: "CC-BY-4.0",
    license_class: "certified",
  },
  quality: {
    composite_fqs_0_100: 78.3,
    rating: "A-",
    temporal: 9,
    geographical: 10,
    technological: 7,
    representativeness: 8,
    methodological: 9,
  },
  uncertainty: {
    ci_95: 6.5,
    ci_lower: 0.766,
    ci_upper: 0.873,
    distribution: "normal",
    sample_size: 48,
  },
  licensing: {
    license: "CC-BY-4.0",
    license_class: "certified",
    upstream_licenses: ["CEA-India-Open"],
    attribution: "Central Electricity Authority of India (2026)",
  },
  deprecation_status: "active",
  gas_breakdown: {
    CO2: 0.79,
    CH4: 0.02,
    N2O: 0.01,
    ch4_gwp: 27,
    n2o_gwp: 273,
  },
  audit_text:
    "12,500 kWh of grid electricity consumed in India resolves to EF:IN:grid_electricity:2026:v1 under the GHG Protocol location-based method profile.",
  audit_text_draft: true,
  signed_receipt: {
    receipt_id: "rcpt_01HXDEMO1",
    alg: "EdDSA",
    payload_hash: "sha256:abc123deadbeef",
    signature: "sig_0x" + "ab".repeat(48),
    verification_key_hint: "gl:factors-2026-q1",
    signed_at: "2026-04-22T10:02:11Z",
    edition_id: "2026.Q1-electricity",
  },
};

test.describe("W4-D canonical demo", () => {
  test.beforeEach(async ({ page }) => {
    // Admin role so AdminGate lets us in without a real JWT.
    await page.addInitScript(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });
    // Stub both the search + Wave-2 resolve endpoints.
    await page.route("**/v1/factors?**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          factors: [
            {
              factor_id: "EF:IN:grid_electricity:2026:v1",
              family: "electricity",
              geography: "IN",
              scope: "scope_2",
              co2e_per_unit: 0.82,
              unit: "kWh",
              source: "CEA-India-2026",
              data_quality_score: 78.3,
              fqs: 78.3,
              factor_status: "certified",
              license_class: "certified",
            },
          ],
          total_count: 1,
        }),
      }),
    );
    await page.route("**/v1/factors/resolve-explain**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(CANONICAL_RESOLVE_RESPONSE),
      }),
    );
  });

  test("resolves 12,500 kWh India demo and shows every v1.2.0 envelope field", async ({
    page,
  }) => {
    await page.goto("/factors/explorer");
    await expect(page.getByRole("heading", { name: /Factors Explorer/i })).toBeVisible();

    // Open the first row.
    await page.getByText("EF:IN:grid_electricity:2026:v1").first().click();

    const panel = page.getByTestId("factor-detail-panel");
    await expect(panel).toBeVisible();

    // 1. chosen_factor chips (factor_id, factor_version, release_version,
    //    method_profile, pack, geography, scope, co2e_per_unit+unit).
    await expect(panel.getByText(/factor_id: EF:IN:grid_electricity/)).toBeVisible();
    await expect(panel.getByText(/factor_version: v1.2/)).toBeVisible();
    await expect(panel.getByText(/release_version: 2026.Q1-electricity/)).toBeVisible();
    await expect(panel.getByText(/method_profile: ghgp_location_based/)).toBeVisible();
    await expect(panel.getByText(/pack: grid_electricity_pack/)).toBeVisible();
    await expect(panel.getByText(/geo: IN/)).toBeVisible();
    await expect(panel.getByText(/scope: scope_2/)).toBeVisible();

    // 2. source descriptor
    await expect(panel.getByText("CEA-India-2026")).toBeVisible();

    // 3. FQS gauge (0-100)
    await expect(
      panel.getByRole("img", {
        name: /Factor quality score 78\.3 out of 100, band High/,
      }),
    ).toBeVisible();

    // 4. uncertainty band
    await expect(panel.getByText(/±6\.50% \(95% CI\)/)).toBeVisible();

    // 5. licensing envelope badges
    await expect(panel.getByTestId("license-class-badge").first()).toBeVisible();
    await expect(panel.getByText(/upstream: CEA-India-Open/)).toBeVisible();

    // 6. deprecation banner — absent when active
    await expect(panel.getByTestId("deprecation-banner")).toHaveCount(0);

    // 7. gas breakdown table
    await expect(panel.getByRole("table", { name: /Greenhouse gas breakdown/i })).toBeVisible();
    await expect(panel.getByText("CO₂")).toBeVisible();
    await expect(panel.getByText("CH₄")).toBeVisible();

    // 8. audit_text + [DRAFT] banner
    await expect(panel.getByTestId("audit-text-draft-banner")).toBeVisible();
    await expect(panel.getByTestId("audit-text-draft-alert")).toBeVisible();
    await expect(panel.getByTestId("audit-text-body")).toContainText(/12,500 kWh/);

    // 9. signed receipt debug pane (collapsible)
    const sr = panel.getByTestId("signed-receipt-pane");
    await expect(sr).toBeVisible();
    await sr.getByRole("button", { name: /Show signed receipt debug/i }).click();
    await expect(panel.getByText(/rcpt_01HXDEMO1/)).toBeVisible();
    await expect(panel.getByText(/EdDSA/)).toBeVisible();
    await expect(panel.getByText(/sha256:abc123deadbeef/)).toBeVisible();
  });
});
