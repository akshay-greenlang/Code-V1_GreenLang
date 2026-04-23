/**
 * AdminGate.evaluateSession unit tests.
 *
 * The gate's redirect behavior is component-level (requires a router
 * context) so it's not covered here; we test the pure session-evaluation
 * logic that decides whether a given token + dev-role combination
 * qualifies as an admin session.
 */
import { describe, expect, it } from "vitest";
import { evaluateSession } from "./AdminGate";

// Produce a minimal unsigned JWT with the given payload. Accepts any
// payload object; signature is irrelevant because AdminGate never
// validates signatures (the API does).
function makeJwt(payload: Record<string, unknown>): string {
  const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${header}.${body}.`;
}

describe("evaluateSession", () => {
  it("treats missing token + non-admin dev role as unauthenticated", () => {
    expect(evaluateSession(null, "operator")).toEqual({
      authenticated: false,
      isAdmin: false,
      reason: "no_token",
    });
  });

  it("honors the dev role switcher when no token is present", () => {
    expect(evaluateSession(null, "admin")).toEqual({
      authenticated: true,
      isAdmin: true,
    });
  });

  it("flags malformed tokens as invalid_token and unauthenticated", () => {
    expect(evaluateSession("garbage", "operator")).toEqual({
      authenticated: false,
      isAdmin: false,
      reason: "invalid_token",
    });
  });

  it("grants admin on role=admin claim", () => {
    const token = makeJwt({ role: "admin" });
    expect(evaluateSession(token, "operator")).toEqual({
      authenticated: true,
      isAdmin: true,
    });
  });

  it("grants admin on factors:admin scope string", () => {
    const token = makeJwt({ role: "user", scope: "profile factors:admin email" });
    expect(evaluateSession(token, "operator")).toEqual({
      authenticated: true,
      isAdmin: true,
    });
  });

  it("grants admin on factors:admin in scope array", () => {
    const token = makeJwt({ role: "user", scopes: ["factors:admin"] });
    expect(evaluateSession(token, "operator")).toEqual({
      authenticated: true,
      isAdmin: true,
    });
  });

  it("denies non-admin authenticated tokens with missing_scope reason", () => {
    const token = makeJwt({ role: "user", scope: "profile email" });
    expect(evaluateSession(token, "operator")).toEqual({
      authenticated: true,
      isAdmin: false,
      reason: "missing_scope",
    });
  });
});
