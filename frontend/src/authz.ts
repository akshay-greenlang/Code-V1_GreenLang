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

export const roleRouteAllowlist: Record<ShellRole, string[]> = {
  operator: [...appRoutes, "/runs"],
  auditor: [...appRoutes, "/runs", "/governance"],
  compliance: [...appRoutes, "/runs", "/governance"],
  admin: [...appRoutes, "/runs", "/governance", "/admin"]
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
