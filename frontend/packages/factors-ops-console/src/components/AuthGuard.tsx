import { useEffect, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { queryKeys } from "@/lib/query";
import {
  canDo,
  fetchIdentity,
  redirectToLogin,
  type Identity,
  type OpsAction,
  type OpsRole,
} from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Wraps a route. If the user is unauthenticated -> redirects to /login.
 * If authenticated but lacks the required role/action -> renders 403.
 *
 * Every route in the Operator Console must be wrapped in <AuthGuard/>; there
 * are no public pages (spec §3.3).
 */
export interface AuthGuardProps {
  children: ReactNode | ((identity: Identity) => ReactNode);
  /** At least one of these roles must be present on the identity. */
  requiredRoles?: OpsRole[];
  /** OR (alternatively) check a named action in the role-action matrix. */
  requiredAction?: OpsAction;
}

export function AuthGuard({
  children,
  requiredRoles,
  requiredAction,
}: AuthGuardProps) {
  const { data: identity, isLoading, isError } = useQuery({
    queryKey: queryKeys.me(),
    queryFn: fetchIdentity,
    staleTime: 60 * 1000,
    retry: false,
  });

  useEffect(() => {
    if (!isLoading && !isError && identity === null) {
      redirectToLogin();
    }
  }, [identity, isLoading, isError]);

  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Authenticating"
        className="flex min-h-[40vh] items-center justify-center text-sm text-muted-foreground"
      >
        Authenticating…
      </div>
    );
  }

  if (!identity) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center text-sm text-muted-foreground">
        Redirecting to sign in…
      </div>
    );
  }

  const roleOk =
    !requiredRoles ||
    requiredRoles.length === 0 ||
    requiredRoles.some((r) => identity.roles.includes(r));
  const actionOk = !requiredAction || canDo(identity, requiredAction);

  if (!roleOk || !actionOk) {
    return <ForbiddenPage required={requiredRoles} identity={identity} />;
  }

  return typeof children === "function" ? <>{children(identity)}</> : <>{children}</>;
}

function ForbiddenPage({
  required,
  identity,
}: {
  required?: OpsRole[];
  identity: Identity;
}) {
  return (
    <div className="mx-auto max-w-lg py-12">
      <Card>
        <CardHeader>
          <CardTitle>403 — Insufficient role</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            You are signed in as <strong>{identity.display_name}</strong> ({identity.email}).
          </p>
          <p className="text-muted-foreground">
            Current roles: {identity.roles.join(", ") || "—"}
          </p>
          {required && required.length > 0 && (
            <p className="text-muted-foreground">
              This screen requires one of: {required.join(", ")}
            </p>
          )}
          <p className="pt-2 text-muted-foreground">
            Ask a GreenLang platform admin to grant the role via Okta.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
