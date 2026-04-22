import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Layers } from "lucide-react";
import { getMethodPacks } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/method-packs")({
  component: MethodPacksIndex,
});

function MethodPacksIndex() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.methodPacks(),
    queryFn: getMethodPacks,
  });

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Method packs</h1>
        <p className="text-sm text-muted-foreground">
          7 opinionated selection policies mapped to standards (GHG Protocol,
          CDP, CBAM, CSRD, etc.). Each pack fully specifies the fallback
          cascade + boundary + GWP basis.
        </p>
      </header>

      {error ? (
        <div
          role="alert"
          className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
        >
          Method packs unavailable.
        </div>
      ) : isLoading ? (
        <div
          className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3"
          aria-busy="true"
        >
          {Array.from({ length: 7 }).map((_, i) => (
            <div
              key={i}
              className="h-36 animate-pulse rounded-lg border border-border bg-muted/40"
            />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {(data ?? []).map((p) => (
            <Card key={p.profile} className="transition-shadow hover:shadow-md">
              <Link
                to="/method-packs/$profile"
                params={{ profile: p.profile }}
                className="block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <CardContent className="space-y-2 p-4">
                  <div className="flex items-center gap-2">
                    <Layers className="h-4 w-4 text-muted-foreground" />
                    <h3 className="truncate font-semibold">{p.name}</h3>
                  </div>
                  <p className="text-xs font-mono text-muted-foreground">
                    {p.profile}
                  </p>
                  <p className="text-sm">{p.purpose}</p>
                  <div className="flex flex-wrap items-center gap-1.5 pt-1">
                    {p.scope_coverage.map((s) => (
                      <Badge key={s} variant="outline">
                        scope {s}
                      </Badge>
                    ))}
                    <Badge variant="secondary">{p.gwp_basis}</Badge>
                  </div>
                  <div className="flex items-center justify-between pt-1 text-xs text-muted-foreground">
                    <span>
                      region hierarchy depth {p.region_hierarchy_depth}
                    </span>
                    <ArrowRight className="h-3 w-3" />
                  </div>
                </CardContent>
              </Link>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
