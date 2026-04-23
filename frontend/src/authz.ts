export type ShellRole = "operator" | "auditor" | "compliance" | "admin";

export const defaultShellRole: ShellRole = "operator";
export const shellRoleStorageKey = "gl.v2.shell.role";

const appRoutes = [
  "/apps/cbam",
  "/apps/csrd",
  "/apps/vcci",
  "/apps/eudr",
  "/apps/ghg",
  "/apps/iso14064",
  "/apps/sb253",
  "/apps/taxonomy"
] as const;

// Public Factors dashboards — available to every role.
const factorsPublicRoutes = ["/factors/status", "/factors/qa"];

// Operator console — admin-only via AdminGate. Listed here so the
// shell-level RoleGuard doesn't bounce admins on its own.
const factorsOperatorRoutes = [
  "/factors/explorer",
  "/factors/sources",
  "/factors/mapping",
  "/factors/diff",
  "/factors/approvals",
  "/factors/overrides",
  "/factors/impact"
];

// Track C-5 OEM white-label onboarding routes. Signup is public; the
// branding + sub-tenants pages are OEM-admin scoped (we keep them on
// the admin allowlist alongside the operator console).
const oemPublicRoutes = ["/oem/signup"];
const oemOperatorRoutes = ["/oem/branding", "/oem/subtenants"];

export const roleRouteAllowlist: Record<ShellRole, string[]> = {
  operator: [...appRoutes, "/runs", ...factorsPublicRoutes, ...oemPublicRoutes],
  auditor: [
    ...appRoutes,
    "/runs",
    "/governance",
    ...factorsPublicRoutes,
    ...oemPublicRoutes
  ],
  compliance: [
    ...appRoutes,
    "/runs",
    "/governance",
    ...factorsPublicRoutes,
    ...oemPublicRoutes
  ],
  admin: [
    ...appRoutes,
    "/runs",
    "/governance",
    "/admin",
    ...factorsPublicRoutes,
    ...factorsOperatorRoutes,
    ...oemPublicRoutes,
    ...oemOperatorRoutes
  ]
};

export function readRoleFromStorage(): ShellRole {
  if (typeof window === "undefined") return defaultShellRole;
  const raw = window.localStorage.getItem(shellRoleStorageKey);
  if (raw === "operator" || raw === "auditor" || raw === "compliance" || raw === "admin") {
    return raw;
  }
  return defaultShellRole;
}

export function writeRoleToStorage(role: ShellRole): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(shellRoleStorageKey, role);
}

export function hasRouteAccess(role: ShellRole, pathname: string): boolean {
  const allowed = roleRouteAllowlist[role];
  return allowed.includes(pathname);
}
