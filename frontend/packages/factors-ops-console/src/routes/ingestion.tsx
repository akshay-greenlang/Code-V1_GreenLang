import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { DownloadCloud, Plus } from "lucide-react";
import { AuthGuard } from "@/components/AuthGuard";
import { IngestionJobRow } from "@/components/IngestionJobRow";
import { ParserLogViewer } from "@/components/ParserLogViewer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { queryKeys } from "@/lib/query";
import { canDo, type Identity } from "@/lib/auth";
import {
  getParserLog,
  listIngestionJobs,
  promoteIngestion,
  rejectIngestion,
  triggerIngest,
} from "@/lib/api";
import type { IngestionJob } from "@/types/ops";

export const Route = createFileRoute("/ingestion")({
  component: IngestionPage,
});

function IngestionPage() {
  return (
    <AuthGuard requiredAction="ingest.run">
      {(identity) => <IngestionConsole identity={identity} />}
    </AuthGuard>
  );
}

function IngestionConsole({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const [openLogFor, setOpenLogFor] = useState<string | null>(null);

  const { data: jobs } = useQuery({
    queryKey: queryKeys.ingestion.jobs(),
    queryFn: listIngestionJobs,
  });
  const { data: log } = useQuery({
    queryKey: openLogFor ? queryKeys.ingestion.log(openLogFor) : ["ingestion", "log", "none"],
    queryFn: () => (openLogFor ? getParserLog(openLogFor) : Promise.resolve([])),
    enabled: Boolean(openLogFor),
  });

  const trigger = useMutation({
    mutationFn: (sourceId: string) =>
      triggerIngest(identity, { source_id: sourceId, reason: "Manual fetch from Ops Console" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.ingestion.jobs() }),
  });
  const promote = useMutation({
    mutationFn: (job: IngestionJob) => promoteIngestion(identity, job.job_id, "Promoted to review queue"),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.ingestion.jobs() }),
  });
  const reject = useMutation({
    mutationFn: (job: IngestionJob) => rejectIngestion(identity, job.job_id, "Rejected: see parser log"),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.ingestion.jobs() }),
  });

  const canMutate = canDo(identity, "ingest.promote");
  const sorted = useMemo(
    () => (jobs ?? []).slice().sort((a, b) => b.started_at.localeCompare(a.started_at)),
    [jobs]
  );

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Ingestion Console</h1>
          <p className="text-sm text-muted-foreground">
            Fetcher jobs, parser logs, and promote/reject actions.
          </p>
        </div>
        {canMutate && (
          <Button
            onClick={() => {
              const src = window.prompt("Source id (e.g. defra_2025)");
              if (src) trigger.mutate(src);
            }}
          >
            <Plus className="h-4 w-4" /> Run new fetcher
          </Button>
        )}
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <DownloadCloud className="h-4 w-4" /> Jobs
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left">
                <tr>
                  <th className="px-3 py-2">job_id</th>
                  <th className="px-3 py-2">source</th>
                  <th className="px-3 py-2">started</th>
                  <th className="px-3 py-2">status</th>
                  <th className="px-3 py-2 text-right">rows</th>
                  <th className="px-3 py-2 text-right">duration</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {sorted.map((j) => (
                  <IngestionJobRow
                    key={j.job_id}
                    job={j}
                    canMutate={canMutate}
                    onPromote={promote.mutate}
                    onReject={reject.mutate}
                  />
                ))}
                {sorted.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-6 text-center text-sm text-muted-foreground">
                      No jobs recorded yet. Trigger a fetcher to begin.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <div>
        <h2 className="mb-2 text-sm font-semibold">Parser log</h2>
        <div className="mb-2 flex flex-wrap gap-1">
          {sorted.map((j) => (
            <button
              key={j.job_id}
              type="button"
              className={`rounded px-2 py-0.5 text-xs ${
                openLogFor === j.job_id
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }`}
              onClick={() => setOpenLogFor(j.job_id)}
            >
              {j.job_id}
            </button>
          ))}
        </div>
        {openLogFor ? (
          <ParserLogViewer entries={log ?? []} />
        ) : (
          <p className="text-xs text-muted-foreground">Pick a job above to view its parser log.</p>
        )}
      </div>
    </div>
  );
}
