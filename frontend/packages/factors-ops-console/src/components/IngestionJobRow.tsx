import { Link } from "@tanstack/react-router";
import { FileText, Play, XCircle, CheckCircle2, Clock, AlertOctagon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDateTime, formatDuration } from "@/lib/utils";
import type { IngestionJob } from "@/types/ops";

interface Props {
  job: IngestionJob;
  onPromote?: (job: IngestionJob) => void;
  onReject?: (job: IngestionJob) => void;
  canMutate: boolean;
}

const STATUS_META: Record<
  IngestionJob["status"],
  { label: string; variant: "success" | "warn" | "danger" | "muted" | "default"; icon: typeof Clock }
> = {
  queued: { label: "queued", variant: "muted", icon: Clock },
  running: { label: "running", variant: "warn", icon: Play },
  completed: { label: "completed", variant: "success", icon: CheckCircle2 },
  failed: { label: "failed", variant: "danger", icon: AlertOctagon },
  promoted: { label: "promoted", variant: "success", icon: CheckCircle2 },
  rejected: { label: "rejected", variant: "danger", icon: XCircle },
};

export function IngestionJobRow({ job, onPromote, onReject, canMutate }: Props) {
  const meta = STATUS_META[job.status];
  const Icon = meta.icon;
  return (
    <tr className="border-t border-border hover:bg-muted/30">
      <td className="px-3 py-2 font-mono text-xs">{job.job_id}</td>
      <td className="px-3 py-2 text-sm">{job.source_label}</td>
      <td className="px-3 py-2 text-xs text-muted-foreground">
        {formatDateTime(job.started_at)}
      </td>
      <td className="px-3 py-2">
        <Badge variant={meta.variant} className="gap-1">
          <Icon className="h-3 w-3" aria-hidden="true" />
          {meta.label}
        </Badge>
      </td>
      <td className="px-3 py-2 text-right tabular-nums">{job.row_count.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-xs text-muted-foreground tabular-nums">
        {formatDuration(job.duration_seconds ?? null)}
      </td>
      <td className="px-3 py-2 text-right">
        <div className="flex items-center justify-end gap-1">
          <Button asChild size="sm" variant="ghost" title="Parser log">
            <Link to="/ingestion" search={{ job: job.job_id } as never}>
              <FileText className="h-3.5 w-3.5" />
              <span className="sr-only">Parser log for {job.job_id}</span>
            </Link>
          </Button>
          {canMutate && job.status === "completed" && onPromote && (
            <Button size="sm" variant="default" onClick={() => onPromote(job)}>
              Promote
            </Button>
          )}
          {canMutate && job.status !== "rejected" && job.status !== "promoted" && onReject && (
            <Button size="sm" variant="outline" onClick={() => onReject(job)}>
              Reject
            </Button>
          )}
        </div>
      </td>
    </tr>
  );
}
