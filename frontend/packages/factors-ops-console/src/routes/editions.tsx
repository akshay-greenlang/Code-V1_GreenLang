import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { EditionPromoter } from "@/components/EditionPromoter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import { getEdition, listEditions, promoteEdition, rollbackEdition } from "@/lib/api";
import type { Identity } from "@/lib/auth";

export const Route = createFileRoute("/editions")({
  component: EditionsPage,
});

function EditionsPage() {
  return (
    <AuthGuard requiredAction="edition.promote">
      {(identity) => <EditionsManagement identity={identity} />}
    </AuthGuard>
  );
}

function EditionsManagement({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const { data: editions } = useQuery({
    queryKey: queryKeys.editions.list(),
    queryFn: listEditions,
  });
  const [focus, setFocus] = useState<string | null>(null);
  const { data: detail } = useQuery({
    queryKey: focus ? queryKeys.editions.detail(focus) : ["editions", "none"],
    queryFn: () => (focus ? getEdition(focus) : Promise.resolve(null)),
    enabled: Boolean(focus),
  });

  const promote = useMutation({
    mutationFn: ({ slice, reason }: { slice: string; reason: string }) =>
      focus ? promoteEdition(identity, focus, slice, reason) : Promise.reject("no edition focused"),
    onSuccess: (updated) => {
      if (focus) qc.setQueryData(queryKeys.editions.detail(focus), updated);
      qc.invalidateQueries({ queryKey: queryKeys.editions.list() });
    },
  });
  const rollback = useMutation({
    mutationFn: (reason: string) =>
      focus ? rollbackEdition(identity, focus, reason) : Promise.reject("no edition focused"),
    onSuccess: (updated) => {
      if (focus) qc.setQueryData(queryKeys.editions.detail(focus), updated);
      qc.invalidateQueries({ queryKey: queryKeys.editions.list() });
    },
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Edition Management</h1>
        <p className="text-sm text-muted-foreground">
          Slice-by-slice promotion with release sign-off checklist. Rollback available.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">All editions</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ul className="divide-y divide-border text-sm">
            {(editions ?? []).map((e) => (
              <li key={e.edition_id}>
                <button
                  type="button"
                  onClick={() => setFocus(e.edition_id)}
                  className={`flex w-full items-center justify-between px-3 py-2 hover:bg-muted/30 ${
                    focus === e.edition_id ? "bg-muted/30" : ""
                  }`}
                >
                  <span className="font-mono text-xs">{e.label}</span>
                  <Badge
                    variant={
                      e.status === "draft"
                        ? "warn"
                        : e.status === "published"
                          ? "success"
                          : "muted"
                    }
                  >
                    {e.status}
                  </Badge>
                </button>
              </li>
            ))}
            {(!editions || editions.length === 0) && (
              <li className="px-3 py-6 text-center text-xs text-muted-foreground">
                No editions recorded.
              </li>
            )}
          </ul>
        </CardContent>
      </Card>

      {detail && (
        <EditionPromoter
          edition={detail}
          onPromote={(slice, reason) => promote.mutateAsync({ slice, reason })}
          onRollback={(reason) => rollback.mutateAsync(reason)}
        />
      )}
    </div>
  );
}
