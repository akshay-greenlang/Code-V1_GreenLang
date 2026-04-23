/**
 * AdminGate — wraps operator-console pages.
 *
 * Permits the child only when the active session is an admin (or holds the
 * `factors:admin` scope). Otherwise redirects to `/login` with a `next`
 * search-param so the login flow can bounce back.
 *
 * Session identity is read from two sources, in order:
 *   1. A JWT in localStorage (`gl.auth.token`) — its `role` and `scope`
 *      claims drive the decision (we parse the unsigned payload locally;
 *      the backend revalidates on every call, so this is purely a UI
 *      hint).
 *   2. The shell's existing role selector (`gl.v2.shell.role` -> "admin")
 *      so that operators using the role switcher in dev still get access
 *      without a real login flow.
 */
import { ReactNode, useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import LinearProgress from "@mui/material/LinearProgress";
import { readRoleFromStorage } from "../../authz";

const TOKEN_STORAGE_KEY = "gl.auth.token";

interface SessionClaims {
  role?: string;
  scope?: string | string[];
  scopes?: string[];
  [k: string]: unknown;
}

function decodeJwtPayload(token: string): SessionClaims | null {
  // Best-effort parse of an unsigned JWT payload. Returns null on any
  // malformed token. We do NOT trust this for authorization — the API
  // re-validates the bearer; this only drives UI-side gating.
  try {
    const parts = token.split(".");
    if (parts.length < 2) return null;
    const payload = parts[1];
    const padLen = (4 - (payload.length % 4)) % 4;
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat(padLen);
    const json =
      typeof atob === "function"
        ? atob(b64)
        : Buffer.from(b64, "base64").toString("utf-8");
    return JSON.parse(json) as SessionClaims;
  } catch {
    return null;
  }
}

export interface SessionEvaluation {
  authenticated: boolean;
  isAdmin: boolean;
  reason?: "no_token" | "invalid_token" | "missing_scope";
}

export function evaluateSession(token: string | null, devRole: string | null): SessionEvaluation {
  if (token) {
    const claims = decodeJwtPayload(token);
    if (!claims) {
      return { authenticated: false, isAdmin: false, reason: "invalid_token" };
    }
    const role = (claims.role ?? "").toString().toLowerCase();
    const scopes: string[] = [];
    if (typeof claims.scope === "string") scopes.push(...claims.scope.split(/\s+/));
    if (Array.isArray(claims.scope)) scopes.push(...claims.scope.map((s) => String(s)));
    if (Array.isArray(claims.scopes)) scopes.push(...claims.scopes.map((s) => String(s)));
    const isAdmin =
      role === "admin" || scopes.includes("factors:admin") || scopes.includes("admin");
    if (!isAdmin) {
      return { authenticated: true, isAdmin: false, reason: "missing_scope" };
    }
    return { authenticated: true, isAdmin: true };
  }
  // No token. Honor the shell's dev role switcher so the existing
  // operator console (used in local dev / demo) still works.
  if (devRole === "admin") {
    return { authenticated: true, isAdmin: true };
  }
  return { authenticated: false, isAdmin: false, reason: "no_token" };
}

interface Props {
  children: ReactNode;
  /** Override redirect target. Default `/login`. */
  loginPath?: string;
}

export function AdminGate({ children, loginPath = "/login" }: Props) {
  const location = useLocation();
  const [verdict, setVerdict] = useState<SessionEvaluation | null>(null);

  useEffect(() => {
    let cancelled = false;
    const run = () => {
      const token =
        typeof window !== "undefined" ? window.localStorage.getItem(TOKEN_STORAGE_KEY) : null;
      const devRole = typeof window !== "undefined" ? readRoleFromStorage() : null;
      if (!cancelled) setVerdict(evaluateSession(token, devRole));
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [location.pathname]);

  if (verdict === null) {
    return (
      <Box sx={{ p: 3 }} role="status" aria-live="polite" aria-label="Verifying admin session">
        <LinearProgress />
      </Box>
    );
  }

  if (!verdict.authenticated) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`${loginPath}?next=${next}`} replace />;
  }

  if (!verdict.isAdmin) {
    return (
      <Box sx={{ p: 3, maxWidth: 720, mx: "auto" }}>
        <Alert severity="error">
          Your session doesn't grant the <code>factors:admin</code> scope. Ask a workspace admin to
          grant access to the operator console.
        </Alert>
      </Box>
    );
  }

  return <>{children}</>;
}

export default AdminGate;
