import { useState } from "react";
import { CheckCircle2, Clock, RotateCcw, ArrowUpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { formatDateTime } from "@/lib/utils";
import type { EditionDetail, EditionSlice } from "@/types/ops";

/**
 * Slice-by-slice release orchestrator (spec §3.4.10).
 *
 * Promote is sequential: you can only promote the first slice whose status is
 * `pending` or `in_review`. Rollback is available once at least one slice has
 * been promoted.
 *
 * Includes a release-signoff checklist gate — the Promote button stays
 * disabled until all checks are ticked. This is a deliberate friction point.
 */
const SIGNOFF_CHECKS: Array<{ id: string; label: string }> = [
  { id: "qa", label: "QA queue clear for this slice" },
  { id: "diff", label: "Cross-edition diff reviewed" },
  { id: "legal", label: "License scanner green" },
  { id: "comms", label: "Release notes drafted" },
];

interface Props {
  edition: EditionDetail;
  onPromote: (slice: string, reason: string) => Promise<void>;
  onRollback: (reason: string) => Promise<void>;
}

export function EditionPromoter({ edition, onPromote, onRollback }: Props) {
  const [reason, setReason] = useState("");
  const [checks, setChecks] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);

  const nextSlice = edition.slices
    .slice()
    .sort((a, b) => a.order - b.order)
    .find((s) => s.status === "pending" || s.status === "in_review");

  const hasPromoted = edition.slices.some((s) => s.status === "promoted");
  const allChecked = SIGNOFF_CHECKS.every((c) => checks[c.id]);
  const reasonOk = reason.trim().length >= 10;
  const canPromote = Boolean(nextSlice) && allChecked && reasonOk && !busy;
  const canRollback = hasPromoted && reasonOk && !busy;

  const toggle = (id: string) =>
    setChecks((prev) => ({ ...prev, [id]: !prev[id] }));

  const handlePromote = async () => {
    if (!nextSlice) return;
    setBusy(true);
    try {
      await onPromote(nextSlice.name, reason);
      setReason("");
      setChecks({});
    } finally {
      setBusy(false);
    }
  };

  const handleRollback = async () => {
    setBusy(true);
    try {
      await onRollback(reason);
      setReason("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2fr_1fr]">
      <Card>
        <CardHeader>
          <CardTitle>
            Edition {edition.label} <Badge variant="outline" className="ml-2">{edition.status}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-1">
            {edition.slices
              .slice()
              .sort((a, b) => a.order - b.order)
              .map((s) => (
                <SliceRow key={s.order} slice={s} />
              ))}
          </ol>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Release sign-off</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <ul className="space-y-1">
            {SIGNOFF_CHECKS.map((c) => (
              <li key={c.id}>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(checks[c.id])}
                    onChange={() => toggle(c.id)}
                  />
                  <span>{c.label}</span>
                </label>
              </li>
            ))}
          </ul>
          <div>
            <label className="text-sm font-medium">Audit reason</label>
            <Textarea
              rows={2}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Release gate passed for Freight slice…"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button disabled={!canPromote} onClick={handlePromote}>
              <ArrowUpCircle className="h-4 w-4" />
              {nextSlice ? `Promote ${nextSlice.name}` : "No next slice"}
            </Button>
            <Button variant="destructive" disabled={!canRollback} onClick={handleRollback}>
              <RotateCcw className="h-4 w-4" /> Rollback edition
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function SliceRow({ slice }: { slice: EditionSlice }) {
  const variant =
    slice.status === "promoted"
      ? "success"
      : slice.status === "in_review"
        ? "warn"
        : slice.status === "rolled_back"
          ? "danger"
          : "muted";
  const Icon = slice.status === "promoted" ? CheckCircle2 : Clock;
  return (
    <li className="flex items-center justify-between rounded-md border border-border px-2 py-1.5">
      <div className="flex items-center gap-2 text-sm">
        <span className="w-5 text-right text-xs text-muted-foreground">{slice.order}.</span>
        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
        <span>{slice.name}</span>
      </div>
      <div className="flex items-center gap-2">
        <Badge variant={variant}>{slice.status.replace("_", " ")}</Badge>
        {slice.by && (
          <span className="text-xs text-muted-foreground">
            {slice.by} · {formatDateTime(slice.at)}
          </span>
        )}
      </div>
    </li>
  );
}
