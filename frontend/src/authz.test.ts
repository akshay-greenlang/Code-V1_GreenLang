import { describe, expect, it } from "vitest";
import { hasRouteAccess } from "./authz";

describe("role route guards", () => {
  it("blocks admin route for operator", () => {
    expect(hasRouteAccess("operator", "/admin")).toBe(false);
  });

  it("allows governance for auditor", () => {
    expect(hasRouteAccess("auditor", "/governance")).toBe(true);
  });

  it("allows admin route for admin role", () => {
    expect(hasRouteAccess("admin", "/admin")).toBe(true);
  });
});
