/**
 * W4-D — Approval queue filter + approve flow.
 *
 * 1. Load queue containing one draft + one approved item.
 * 2. Flip the "Drafts only" filter — approved item disappears.
 * 3. Approve the draft — banner chip flips from [DRAFT] to approved after
 *    the queue reloads.
 */
import { expect, test } from "@playwright/test";

let approveCount = 0;

const DRAFT_ITEM = {
  review_id: "rev-draft-1",
  factor_id: "EF:IN:grid_electricity:2026:v1",
  family: "electricity",
  current_status: "preview",
  proposed_status: "certified",
  submitted_by: "alice@example.com",
  submitted_at: "2026-04-20T12:00:00Z",
  rationale: "CEA India 2026 refresh",
  reviewer: null,
  due_date: null,
  evidence: {
    audit_text: "Draft narrative pending methodology sign-off.",
    audit_text_draft: true,
  },
};

const APPROVED_ITEM = {
  review_id: "rev-approved-1",
  factor_id: "EF:UK:road_freight:2026:v1",
  family: "freight",
  current_status: "draft",
  proposed_status: "preview",
  submitted_by: "bob@example.com",
  submitted_at: "2026-04-21T08:00:00Z",
  rationale: "DEFRA 2026 refresh",
  reviewer: "carol@example.com",
  due_date: null,
  evidence: {
    audit_text: "Approved narrative.",
    audit_text_draft: false,
  },
};

test.describe("W4-D approval queue", () => {
  test.beforeEach(async ({ page }) => {
    approveCount = 0;
    await page.addInitScript(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });
    await page.route("**/v1/admin/queue", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items: approveCount > 0 ? [APPROVED_ITEM] : [DRAFT_ITEM, APPROVED_ITEM],
        }),
      }),
    );
    await page.route("**/v1/admin/queue/*/approve", (route) => {
      approveCount += 1;
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true, new_status: "certified" }),
      });
    });
  });

  test("drafts filter hides approved rows; approve removes the draft banner", async ({ page }) => {
    await page.goto("/factors/approvals");
    await expect(page.getByRole("heading", { name: /Approval Queue/i })).toBeVisible();

    // Both chips visible initially.
    await expect(page.getByTestId("audit-text-chip-draft")).toBeVisible();
    await expect(page.getByTestId("audit-text-chip-approved")).toBeVisible();

    // Toggle drafts-only.
    await page.getByTestId("filter-draft-only").click();
    await expect(page.getByTestId("audit-text-chip-approved")).toHaveCount(0);
    await expect(page.getByTestId("audit-text-chip-draft")).toBeVisible();

    // Approve the draft row (first Approve button).
    await page.getByRole("button", { name: /^Approve$/ }).first().click();

    // After reload, the draft chip should be gone; the approved chip
    // appears once the filter is switched off.
    await page.getByTestId("filter-draft-only").click();
    await expect(page.getByTestId("audit-text-chip-draft")).toHaveCount(0);
    await expect(page.getByTestId("audit-text-chip-approved")).toBeVisible();
  });
});
