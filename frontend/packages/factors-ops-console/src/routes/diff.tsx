import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { DiffTable } from "@/components/DiffTable";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { getDiff, listEditions } from "@/lib/api";
import { queryKeys } from "@/lib/query";

export const Route = createFileRoute("/diff")({
  component: DiffPage,
});

function DiffPage() {
  return (
    <AuthGuard>
      <DiffViewer />
    </AuthGuard>
  );
}

function DiffViewer() {
  const { data: editions } = useQuery({
    queryKey: queryKeys.editions.list(),
    queryFn: listEditions,
  });
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [focusFactor, setFocusFactor] = useState("");

  const { data: diffs, isFetching } = useQuery({
    queryKey: left && right ? queryKeys.diff(left, right) : ["diff", "none"],
    queryFn: () => getDiff(left, right),
    enabled: Boolean(left && right),
  });

  const focused = focusFactor
    ? (diffs ?? []).find((d) => d.factor_id === focusFactor)
    : undefined;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Diff Viewer</h1>
        <p className="text-sm text-muted-foreground">
          Pick two editions; drill into a factor for field-level diff.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Choose editions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Left</label>
              <select
                value={left}
                onChange={(e) => setLeft(e.target.value)}
                className="mt-1 h-9 rounded-md border border-border bg-background px-2 text-sm"
              >
                <option value="">—</option>
                {(editions ?? []).map((e) => (
                  <option key={e.edition_id} value={e.edition_id}>
                    {e.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Right</label>
              <select
                value={right}
                onChange={(e) => setRight(e.target.value)}
                className="mt-1 h-9 rounded-md border border-border bg-background px-2 text-sm"
              >
                <option value="">—</option>
                {(editions ?? []).map((e) => (
                  <option key={e.edition_id} value={e.edition_id}>
                    {e.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex-1 min-w-[14rem]">
              <label className="text-xs text-muted-foreground">Focus factor id</label>
              <Input
                value={focusFactor}
                onChange={(e) => setFocusFactor(e.target.value)}
                placeholder="DEFRA-ELEC-GB-2025-001"
                className="font-mono"
              />
            </div>
            <Button variant="ghost" onClick={() => { setLeft(""); setRight(""); setFocusFactor(""); }}>
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      {left && right ? (
        <div className="space-y-4">
          {isFetching && (
            <p className="text-sm text-muted-foreground">Loading diff…</p>
          )}
          {focused ? (
            <DiffTable diff={focused} editable={false} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {left} → {right}
                  <span className="ml-2 text-xs font-normal text-muted-foreground">
                    {diffs?.length ?? 0} changed factors
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 text-left">
                      <tr>
                        <th className="px-3 py-2">factor_id</th>
                        <th className="px-3 py-2">status</th>
                        <th className="px-3 py-2">fields changed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(diffs ?? []).map((d) => (
                        <tr
                          key={d.factor_id}
                          className="cursor-pointer border-t border-border hover:bg-muted/30"
                          onClick={() => setFocusFactor(d.factor_id)}
                        >
                          <td className="px-3 py-2 font-mono text-xs">{d.factor_id}</td>
                          <td className="px-3 py-2 text-xs">{d.status}</td>
                          <td className="px-3 py-2 text-xs">{d.changes.length}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          Select a left and right edition to load the diff.
        </p>
      )}
    </div>
  );
}
