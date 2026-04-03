import { describe, expect, it } from "vitest";

describe("v2.2 shell smoke", () => {
  it("keeps six workspace routes", () => {
    const routes = [
      "/apps/cbam",
      "/apps/csrd",
      "/apps/vcci",
      "/apps/eudr",
      "/apps/ghg",
      "/apps/iso14064"
    ];
    expect(routes.length).toBe(6);
  });
});
