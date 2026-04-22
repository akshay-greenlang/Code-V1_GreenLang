import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { MappingEditor } from "@/components/MappingEditor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { queryKeys } from "@/lib/query";
import { listMappings, getMapping, saveMapping } from "@/lib/api";
import type { Identity } from "@/lib/auth";
import type { MappingRow, MappingSet } from "@/types/ops";

export const Route = createFileRoute("/mapping")({
  component: MappingPage,
});

function MappingPage() {
  return (
    <AuthGuard requiredAction="mapping.edit">
      {(identity) => <MappingWorkbench identity={identity} />}
    </AuthGuard>
  );
}

function MappingWorkbench({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reason, setReason] = useState("");

  const { data: sets } = useQuery({
    queryKey: queryKeys.mapping.sets(),
    queryFn: listMappings,
  });
  const { data: active } = useQuery({
    queryKey: selectedId ? queryKeys.mapping.set(selectedId) : ["mapping", "none"],
    queryFn: () => (selectedId ? getMapping(selectedId) : Promise.resolve(null)),
    enabled: Boolean(selectedId),
  });

  const save = useMutation({
    mutationFn: async ({ id, rows }: { id: string; rows: MappingRow[] }) =>
      saveMapping(identity, id, { rows, reason }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.mapping.sets() });
      setReason("");
    },
  });

  const [draftRows, setDraftRows] = useState<MappingRow[] | null>(null);
  const rows: MappingRow[] = draftRows ?? (active?.rows ?? []);

  const updateRow = (idx: number, patch: Partial<MappingRow>) => {
    const base = draftRows ?? (active?.rows ?? []);
    setDraftRows(base.map((r) => (r.index === idx ? { ...r, ...patch } : r)));
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Mapping Workbench</h1>
        <p className="text-sm text-muted-foreground">
          Map raw activity-text to canonical factors with suggestion-agent assist.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Mapping sets</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-1 text-xs">
          {(sets ?? []).map((s) => (
            <button
              key={s.mapping_set_id}
              type="button"
              className={`rounded-md border border-border px-2 py-1 ${
                selectedId === s.mapping_set_id ? "bg-primary text-primary-foreground" : ""
              }`}
              onClick={() => {
                setSelectedId(s.mapping_set_id);
                setDraftRows(null);
              }}
            >
              {s.name} · {s.unmapped_count}/{s.total_count} unmapped
            </button>
          ))}
          {(!sets || sets.length === 0) && (
            <span className="text-muted-foreground">No mapping sets.</span>
          )}
        </CardContent>
      </Card>

      {active && <ActiveMappingPanel
        set={active as MappingSet}
        rows={rows}
        onAccept={(idx, fid) => updateRow(idx, { state: "accepted", accepted: fid })}
        onReject={(idx) => updateRow(idx, { state: "rejected" })}
        onManualPick={(idx, fid) => updateRow(idx, { state: "accepted", accepted: fid })}
        reason={reason}
        onReasonChange={setReason}
        onSave={() => save.mutate({ id: active.mapping_set_id, rows })}
        canSubmit={reason.trim().length >= 10}
      />}
    </div>
  );
}

function ActiveMappingPanel({
  set,
  rows,
  onAccept,
  onReject,
  onManualPick,
  reason,
  onReasonChange,
  onSave,
  canSubmit,
}: {
  set: MappingSet;
  rows: MappingRow[];
  onAccept: (idx: number, factorId: string) => void;
  onReject: (idx: number) => void;
  onManualPick: (idx: number, factorId: string) => void;
  reason: string;
  onReasonChange: (v: string) => void;
  onSave: () => void;
  canSubmit: boolean;
}) {
  const unmapped = rows.filter((r) => r.state === "unmapped" || r.state === "suggested").length;
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {set.name}{" "}
          <span className="ml-2 text-xs text-muted-foreground">
            {unmapped} unmapped / {set.total_count} total
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <MappingEditor
          rows={rows}
          onAccept={onAccept}
          onReject={onReject}
          onManualPick={onManualPick}
        />
        <div className="space-y-1">
          <label className="text-sm font-medium">Audit reason (required)</label>
          <Textarea
            value={reason}
            onChange={(e) => onReasonChange(e.target.value)}
            rows={2}
            placeholder="Mapped all unmapped rows per QA criteria v4"
          />
        </div>
        <Button disabled={!canSubmit} onClick={onSave}>
          Save draft & submit for approval
        </Button>
      </CardContent>
    </Card>
  );
}
