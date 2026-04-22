import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { NavSidebar } from "@/components/NavSidebar";
import { RoleBadge } from "@/components/RoleBadge";

/**
 * Root route: wraps the whole app in <AuthGuard/> (spec §3.3 — no public
 * pages). Renders OperatorShell: sidebar + top bar + <Outlet/>.
 */
export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootShell,
});

function RootShell() {
  return (
    <AuthGuard>
      {(identity) => (
        <div className="flex min-h-full bg-background">
          <NavSidebar identity={identity} />
          <div className="flex min-h-full flex-1 flex-col">
            <header className="sticky top-0 z-10 flex h-12 items-center justify-between border-b border-border bg-background/95 px-4 backdrop-blur">
              <span className="text-xs uppercase tracking-wider text-muted-foreground">
                Factors Ops Console · internal
              </span>
              <RoleBadge identity={identity} />
            </header>
            <main className="flex-1 p-6">
              <Outlet />
            </main>
          </div>
        </div>
      )}
    </AuthGuard>
  );
}
