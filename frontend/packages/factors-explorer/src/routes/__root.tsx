import { createRootRouteWithContext } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/Layout";

/** Root route carrying the QueryClient on context. */
export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()(
  {
    component: Layout,
  }
);
