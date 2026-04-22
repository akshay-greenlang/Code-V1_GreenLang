import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { AuthGuard } from "@/components/AuthGuard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import { classifyWatchEvent, listWatchEvents } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import type { Identity } from "@/lib/auth";
import type { WatchEvent } from "@/types/ops";

export const Route = createFileRoute("/watch")({
  component: WatchPage,
});

function WatchPage() {
  return (
    <AuthGuard requiredAction="watch.classify">
      {(identity) => <SourceWatch identity={identity} />}
    </AuthGuard>
  );
}

function SourceWatch({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: queryKeys.watch.events(),
    queryFn: listWatchEvents,
  });
  const classify = useMutation({
    mutationFn: ({
      id,
      classification,
    }: {
      id: string;
      classification: "major" | "minor" | "patch" | "noop";
    }) =>
      classifyWatchEvent(
        identity,
        id,
        classification,
        `Classified as ${classification} from Source Watch`
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.watch.events() }),
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Source Watch</h1>
        <p className="text-sm text-muted-foreground">
          Pending source-side changes — detected by source_watch/change_detector; review and classify.
        </p>
      </header>

      <div className="space-y-3">
        {(data ?? []).map((ev) => (
          <SourceWatchCard
            key={ev.detection_id}
            event={ev}
            onClassify={(c) => classify.mutate({ id: ev.detection_id, classification: c })}
          />
        ))}
        {(!data || data.length === 0) && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No source-side changes pending.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function SourceWatchCard({
  event,
  onClassify,
}: {
  event: WatchEvent;
  onClassify: (c: "major" | "minor" | "patch" | "noop") => void;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm">
          <span>
            <code className="font-mono">{event.source_label}</code>{" "}
            <Badge variant="warn" className="ml-2">
              {event.signal.replace("_", " ")}
            </Badge>
          </span>
          <span className="text-xs text-muted-foreground">
            detected {formatDateTime(event.detected_at)}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center justify-between gap-3">
        {event.doc_diff_url ? (
          <a
            href={event.doc_diff_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-primary underline-offset-2 hover:underline"
          >
            <ExternalLink className="h-3 w-3" /> doc_diff view
          </a>
        ) : (
          <span className="text-xs text-muted-foreground">No doc_diff available.</span>
        )}
        <div className="flex flex-wrap gap-1">
          {event.classified && event.classification ? (
            <Badge variant="success">classified {event.classification}</Badge>
          ) : (
            (["major", "minor", "patch", "noop"] as const).map((c) => (
              <Button key={c} size="sm" variant="outline" onClick={() => onClassify(c)}>
                {c}
              </Button>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
