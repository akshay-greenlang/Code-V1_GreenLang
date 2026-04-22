import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { getSource } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LicenseBadge } from "@/components/LicenseBadge";
import { Button } from "@/components/ui/button";
import { formatDate } from "@/lib/utils";

export const Route = createFileRoute("/sources/$sourceId")({
  component: SourceDetailPage,
});

function SourceDetailPage() {
  const { sourceId } = Route.useParams();

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.source(sourceId),
    queryFn: () => getSource(sourceId),
  });

  if (isLoading) {
    return (
      <div
        aria-busy="true"
        className="h-64 animate-pulse rounded-lg border border-border bg-muted/40"
      />
    );
  }

  if (error || !data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
      >
        Source <code className="font-mono">{sourceId}</code> not found.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="font-mono text-xs text-muted-foreground">
                {data.source_id}
              </p>
              <CardTitle className="mt-1">{data.name}</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground">
                {data.publisher}
                {data.jurisdiction ? ` • ${data.jurisdiction}` : ""}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <LicenseBadge licenseClass={data.license_class} />
              <span className="rounded-full bg-muted px-2 py-1 font-mono text-xs">
                v{data.current_version}
              </span>
              <span className="rounded-full bg-muted px-2 py-1 text-xs">
                {data.cadence}
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {data.description ? (
            <p className="text-muted-foreground">{data.description}</p>
          ) : null}
          <div className="grid gap-2 md:grid-cols-3">
            <Stat label="Factors" value={data.factor_count.toLocaleString()} />
            <Stat
              label="Validity"
              value={`${formatDate(data.validity_start)}${data.validity_end ? ` → ${formatDate(data.validity_end)}` : ""}`}
            />
            <Stat label="Last updated" value={formatDate(data.last_updated)} />
          </div>
          {data.jurisdiction_coverage.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1 text-xs">
              <span className="text-muted-foreground">Coverage:</span>
              {data.jurisdiction_coverage.map((j) => (
                <span
                  key={j}
                  className="rounded bg-muted px-1.5 py-0.5 font-mono"
                >
                  {j}
                </span>
              ))}
            </div>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline" size="sm">
              <Link
                to="/search"
                search={{
                  q: "",
                  source_id: data.source_id,
                  offset: 0,
                  limit: 20,
                }}
              >
                Browse {data.factor_count.toLocaleString()} factors
              </Link>
            </Button>
            {data.url ? (
              <Button asChild variant="outline" size="sm">
                <a href={data.url} target="_blank" rel="noopener noreferrer">
                  Publisher site <ExternalLink className="h-3 w-3" />
                </a>
              </Button>
            ) : null}
          </div>
        </CardContent>
      </Card>

      {data.changelog.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Changelog</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2 text-sm">
              {data.changelog.map((c) => (
                <li
                  key={`${c.version}-${c.date}`}
                  className="border-l-2 border-border pl-3"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs">v{c.version}</span>
                    <span className="text-xs text-muted-foreground">
                      {formatDate(c.date)}
                    </span>
                  </div>
                  <p>{c.summary}</p>
                  {c.diff_url ? (
                    <a
                      href={c.diff_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs underline underline-offset-2"
                    >
                      view diff
                    </a>
                  ) : null}
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <p className="tabular-nums">{value}</p>
    </div>
  );
}
