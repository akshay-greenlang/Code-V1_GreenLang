# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: fixture-runs.spec.ts >> E2E fixture runs (GL_SHELL_E2E_FIXTURES) >> Run Center shows fixture run after blocked fixture POST
- Location: e2e\fixture-runs.spec.ts:33:3

# Error details

```
Test timeout of 30000ms exceeded.
```

```
Error: page.goto: Test timeout of 30000ms exceeded.
Call log:
  - navigating to "http://127.0.0.1:4179/runs", waiting until "load"

```

# Test source

```ts
  1  | import { expect, test } from "@playwright/test";
  2  | 
  3  | test.describe("E2E fixture runs (GL_SHELL_E2E_FIXTURES)", () => {
  4  |   test("blocked fixture returns FAIL chip and blocked state", async ({ request }) => {
  5  |     const res = await request.post("/api/v1/e2e/shell-fixture-run", {
  6  |       data: { mode: "blocked" },
  7  |       headers: { "Content-Type": "application/json" }
  8  |     });
  9  |     expect(res.ok(), await res.text()).toBeTruthy();
  10 |     const body = (await res.json()) as {
  11 |       run_state?: string;
  12 |       status_chip?: string;
  13 |       lifecycle_phase?: string;
  14 |       error_envelope?: { message?: string } | null;
  15 |     };
  16 |     expect(body.run_state).toBe("blocked");
  17 |     expect(body.status_chip).toBe("FAIL");
  18 |     expect(body.lifecycle_phase).toBe("completed");
  19 |     expect(body.error_envelope?.message).toBeTruthy();
  20 |   });
  21 | 
  22 |   test("partial_success fixture returns WARN chip", async ({ request }) => {
  23 |     const res = await request.post("/api/v1/e2e/shell-fixture-run", {
  24 |       data: { mode: "partial_success" },
  25 |       headers: { "Content-Type": "application/json" }
  26 |     });
  27 |     expect(res.ok(), await res.text()).toBeTruthy();
  28 |     const body = (await res.json()) as { run_state?: string; status_chip?: string };
  29 |     expect(body.run_state).toBe("partial_success");
  30 |     expect(body.status_chip).toBe("WARN");
  31 |   });
  32 | 
  33 |   test("Run Center shows fixture run after blocked fixture POST", async ({ page, request }) => {
  34 |     const res = await request.post("/api/v1/e2e/shell-fixture-run", {
  35 |       data: { mode: "blocked" },
  36 |       headers: { "Content-Type": "application/json" }
  37 |     });
  38 |     expect(res.ok()).toBeTruthy();
  39 |     const body = (await res.json()) as { run_id?: string };
  40 |     expect(body.run_id).toBeTruthy();
  41 | 
> 42 |     await page.goto("/runs");
     |                ^ Error: page.goto: Test timeout of 30000ms exceeded.
  43 |     await page.getByRole("heading", { name: /Run Center/i }).waitFor();
  44 |     await expect(page.getByText(body.run_id!)).toBeVisible({ timeout: 15_000 });
  45 |   });
  46 | });
  47 | 
```