import { createFileRoute, Link } from "@tanstack/react-router";
import { useQueries } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import {
  listQaFailures,
  listReviewQueue,
  listWatchEvents,
  listEditions,
} from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

export const Route = createFileRoute("/")({
  component: OpsDashboard,
});

function OpsDashboard() {
  const [reviews, qa, watch, editions] = useQueries({
    queries: [
      { queryKey: queryKeys.reviews.queue(), queryFn: listReviewQueue },
      { queryKey: queryKeys.qa.failures(null), queryFn: () => listQaFailures() },
      { queryKey: queryKeys.watch.events(), queryFn: listWatchEvents },
      { queryKey: queryKeys.editions.list(), queryFn: listEditions },
    ],
  });

  const currentEdition = (editions.data ?? []).find((e) => e.status === "draft")
    ?? (editions.data ?? []).find((e) => e.status === "published");
  const latestWatch = (watch.data ?? [])
    .slice()
    .sort((a, b) => b.detected_at.localeCompare(a.detected_at))[0];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Ops dashboard</h1>
        <p className="text-sm text-muted-foreground">
          At-a-glance state of the methodology pipeline.
        </p>
      </header>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          title="Open reviews"
          value={reviews.data?.length ?? "—"}
          href="/approvals"
          hint="pending methodology approvals"
        />
        <KpiCard
          title="QA failures"
          value={qa.data?.length ?? "—"}
          href="/qa"
          hint="validators / dedup / cross_source / license"
          variant={(qa.data?.length ?? 0) > 0 ? "warn" : "muted"}
        />
        <KpiCard
          title="Pending approvals"
          value={
            reviews.data?.filter((r) => r.steps.some((s) => s.status === "pending")).length ?? "—"
          }
          href="/approvals"
          hint="awaiting signoff"
        />
        <KpiCard
          title="Current edition"
          value={currentEdition?.label ?? "—"}
          href="/editions"
          hint={currentEdition ? currentEdition.status : "no draft"}
          variant={currentEdition?.status === "draft" ? "warn" : "success"}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Latest source watch signal</CardTitle>
        </CardHeader>
        <CardContent className="text-sm">
          {latestWatch ? (
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Badge variant="warn">{latestWatch.signal}</Badge>
                <span className="font-mono">{latestWatch.source_label}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                detected {formatDateTime(latestWatch.detected_at)}
              </div>
              <Link to="/watch" className="text-xs text-primary underline-offset-2 hover:underline">
                Open Source Watch →
              </Link>
            </div>
          ) : (
            <div className="text-muted-foreground">No pending source-side changes.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function KpiCard({
  title,
  value,
  href,
  hint,
  variant = "default",
}: {
  title: string;
  value: string | number;
  href: string;
  hint?: string;
  variant?: "default" | "success" | "warn" | "danger" | "muted";
}) {
  return (
    <Link to={href} className="block">
      <Card className="transition-colors hover:bg-muted/30">
        <CardHeader className="pb-1">
          <CardTitle className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{title}</span>
            <Badge variant={variant}>{variant}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-semibold tabular-nums">{value}</div>
          {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
        </CardContent>
      </Card>
    </Link>
  );
}
