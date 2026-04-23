/**
 * W4-D — Audit bundle export flow.
 *
 * 1. Submit an export job.
 * 2. Poll until the stub flips status to "completed".
 * 3. Assert the download button appears with the correct href.
 */
import { expect, test } from "@playwright/test";

let pollCount = 0;

const JOB_ID = "bundle-job-42";

test.describe("W4-D audit bundle export", () => {
  test.beforeEach(async ({ page }) => {
    pollCount = 0;
    await page.addInitScript(() => {
      window.localStorage.setItem("gl.v2.shell.role", "admin");
    });

    await page.route("**/v1/admin/audit-bundles/export", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: JOB_ID,
          status: "running",
          created_at: "2026-04-22T10:00:00Z",
          included_factors: 1287,
        }),
      }),
    );
    await page.route(`**/v1/admin/audit-bundles/jobs/${JOB_ID}`, (route) => {
      pollCount += 1;
      if (pollCount >= 2) {
        route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            job_id: JOB_ID,
            status: "completed",
            created_at: "2026-04-22T10:00:00Z",
            completed_at: "2026-04-22T10:00:05Z",
            included_factors: 1287,
            download_url: "https://bundles.example.com/audit/bundle-job-42.zip",
          }),
        });
      } else {
        route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            job_id: JOB_ID,
            status: "running",
            created_at: "2026-04-22T10:00:00Z",
            included_factors: 1287,
          }),
        });
      }
    });
  });

  test("submits an export job and surfaces the download", async ({ page }) => {
    await page.goto("/factors/audit-bundles");

    await page.getByTestId("audit-bundle-factor-ids").fill(
      "EF:IN:grid_electricity:2026:v1\nEF:UK:road_freight_40t:2026:v1",
    );
    await page.getByTestId("audit-bundle-submit").click();

    await expect(page.getByTestId("audit-bundle-job-panel")).toBeVisible();
    await expect(page.getByTestId("audit-bundle-job-status")).toHaveText(/running|completed/);

    // Wait for the poll to flip status.
    await expect(page.getByTestId("audit-bundle-job-status")).toHaveText("completed", {
      timeout: 10_000,
    });
    const download = page.getByTestId("audit-bundle-download");
    await expect(download).toBeVisible();
    await expect(download).toHaveAttribute(
      "href",
      "https://bundles.example.com/audit/bundle-job-42.zip",
    );
  });
});
