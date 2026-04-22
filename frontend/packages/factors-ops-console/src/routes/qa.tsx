import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { ValidationFailureRow } from "@/components/ValidationFailureRow";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { canDo, type Identity } from "@/lib/auth";
import { listQaFailures, resolveQaItem } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import type { ValidationFailure } from "@/types/ops";

export const Route = createFileRoute("/qa")({
  component: QaPage,
});

const MODULES = ["all", "validators", "dedup_engine", "cross_source", "license_scanner"] as const;

function QaPage() {
  return (
    <AuthGuard requiredRoles={["admin", "data_curator", "reviewer", "viewer"]}>
      {(identity) => <QaDashboard identity={identity} />}
    </AuthGuard>
  );
}

function QaDashboard({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const [filter, setFilter] = useState<(typeof MODULES)[number]>("all");
  const { data: failures } = useQuery({
    queryKey: queryKeys.qa.failures(filter === "all" ? null : filter),
    queryFn: () => listQaFailures(filter === "all" ? null : filter),
  });

  const remediate = useMutation({
    mutationFn: ({
      f,
      resolution,
    }: {
      f: ValidationFailure;
      resolution: "approve" | "reject" | "fix";
    }) =>
      resolveQaItem(identity, f.id, {
        resolution,
        reason: `QA ${resolution} — ${f.message.slice(0, 80)}`,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.qa.failures(null) }),
  });

  const canRemediate = canDo(identity, "qa.remediate");

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">QA Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Failures from validators, dedup_engine, cross_source, and license_scanner.
        </p>
      </header>

      <div className="flex flex-wrap gap-1 text-xs">
        {MODULES.map((m) => (
          <button
            key={m}
            type="button"
            className={`rounded-md border border-border px-2 py-1 ${
              filter === m ? "bg-primary text-primary-foreground" : ""
            }`}
            onClick={() => setFilter(m)}
          >
            {m.replace("_", " ")}
          </button>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            QA Failures ({failures?.length ?? 0} open)
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left">
                <tr>
                  <th className="px-3 py-2">id</th>
                  <th className="px-3 py-2">module</th>
                  <th className="px-3 py-2">severity</th>
                  <th className="px-3 py-2">factor_id</th>
                  <th className="px-3 py-2">message</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {(failures ?? []).map((f) => (
                  <ValidationFailureRow
                    key={f.id}
                    failure={f}
                    canRemediate={canRemediate}
                    onRemediate={(ff, resolution) => remediate.mutate({ f: ff, resolution })}
                  />
                ))}
                {(!failures || failures.length === 0) && (
                  <tr>
                    <td colSpan={6} className="px-3 py-6 text-center text-sm text-muted-foreground">
                      No failures in this scope. All green.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
