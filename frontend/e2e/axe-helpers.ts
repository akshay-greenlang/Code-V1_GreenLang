import AxeBuilder from "@axe-core/playwright";
import { expect } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * Fails the test if axe reports serious or critical violations.
 * Color-contrast is included by default; set `includeColorContrast: false` only while triaging.
 */
export async function assertNoSeriousAxeViolations(
  page: Page,
  options?: { includeColorContrast?: boolean }
): Promise<void> {
  let builder = new AxeBuilder({ page });
  if (options?.includeColorContrast === false) {
    builder = builder.disableRules(["color-contrast"]);
  }
  const results = await builder.analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious, JSON.stringify(serious, null, 2)).toHaveLength(0);
}
