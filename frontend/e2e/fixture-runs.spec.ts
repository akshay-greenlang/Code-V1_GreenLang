import { expect, test } from "@playwright/test";

test.describe("E2E fixture runs (GL_SHELL_E2E_FIXTURES)", () => {
  test("blocked fixture returns FAIL chip and blocked state", async ({ request }) => {
    const res = await request.post("/api/v1/e2e/shell-fixture-run", {
      data: { mode: "blocked" },
      headers: { "Content-Type": "application/json" }
    });
    expect(res.ok(), await res.text()).toBeTruthy();
    const body = (await res.json()) as {
      run_state?: string;
      status_chip?: string;
      lifecycle_phase?: string;
      error_envelope?: { message?: string } | null;
    };
    expect(body.run_state).toBe("blocked");
    expect(body.status_chip).toBe("FAIL");
    expect(body.lifecycle_phase).toBe("completed");
    expect(body.error_envelope?.message).toBeTruthy();
  });

  test("partial_success fixture returns WARN chip", async ({ request }) => {
    const res = await request.post("/api/v1/e2e/shell-fixture-run", {
      data: { mode: "partial_success" },
      headers: { "Content-Type": "application/json" }
    });
    expect(res.ok(), await res.text()).toBeTruthy();
    const body = (await res.json()) as { run_state?: string; status_chip?: string };
    expect(body.run_state).toBe("partial_success");
    expect(body.status_chip).toBe("WARN");
  });

});
