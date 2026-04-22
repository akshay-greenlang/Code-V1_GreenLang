import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getMethodPack } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/method-packs/$profile")({
  component: MethodPackDetailPage,
});

function MethodPackDetailPage() {
  const { profile } = Route.useParams();

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.methodPack(profile),
    queryFn: () => getMethodPack(profile),
  });

  if (isLoading)
    return (
      <div
        aria-busy="true"
        className="h-64 animate-pulse rounded-lg border border-border bg-muted/40"
      />
    );

  if (error || !data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
      >
        Method pack <code className="font-mono">{profile}</code> not found.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <p className="font-mono text-xs text-muted-foreground">
            {data.profile}
          </p>
          <CardTitle>{data.name}</CardTitle>
          <p className="text-sm text-muted-foreground">{data.purpose}</p>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2 text-xs">
          {data.scope_coverage.map((s) => (
            <Badge key={s} variant="outline">
              scope {s}
            </Badge>
          ))}
          <Badge variant="secondary">GWP basis: {data.gwp_basis}</Badge>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Selection rules</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2 text-sm">
              {data.selection_rules.map((r) => (
                <li key={r.rank} className="flex gap-2">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-muted font-mono text-xs">
                    {r.rank}
                  </span>
                  <div>
                    <p>{r.rule}</p>
                    {r.example ? (
                      <p className="mt-0.5 text-xs italic text-muted-foreground">
                        e.g. {r.example}
                      </p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Boundary rules</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-inside list-disc space-y-1 text-sm">
              {data.boundary_rules.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Region hierarchy</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="flex flex-wrap items-center gap-1 text-sm">
              {data.region_hierarchy.map((r, i) => (
                <li key={`${r}-${i}`} className="flex items-center gap-1">
                  <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs">
                    {r}
                  </span>
                  {i < data.region_hierarchy.length - 1 ? (
                    <span className="text-muted-foreground">→</span>
                  ) : null}
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Fallback logic</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-1 text-sm">
              {data.fallback_logic.map((f) => (
                <li key={f.rank} className="flex gap-2">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-muted font-mono text-xs">
                    {f.rank}
                  </span>
                  <div>
                    <p className="font-mono text-xs">{f.step_label}</p>
                    <p className="text-muted-foreground">{f.description}</p>
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>

      {data.reporting_labels.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Reporting labels</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1 text-sm">
              {data.reporting_labels.map((l, i) => (
                <li key={i} className="flex items-center gap-2">
                  <Badge variant="outline">{l.standard}</Badge>
                  <span>{l.label}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
