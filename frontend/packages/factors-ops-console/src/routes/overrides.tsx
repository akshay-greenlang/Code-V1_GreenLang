import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { OverrideEditor } from "@/components/OverrideEditor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import { listOverrides, saveOverride } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type { Identity } from "@/lib/auth";

export const Route = createFileRoute("/overrides")({
  component: OverridesPage,
});

function OverridesPage() {
  return (
    <AuthGuard requiredAction="override.edit">
      {(identity) => <OverridesManager identity={identity} />}
    </AuthGuard>
  );
}

function OverridesManager({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const [tenantId, setTenantId] = useState<string>(identity.tenant_id);
  const [factorId, setFactorId] = useState<string>("");
  const { data: overrides } = useQuery({
    queryKey: queryKeys.tenant(tenantId).overrides(),
    queryFn: () => listOverrides(identity, tenantId),
    enabled: Boolean(tenantId),
  });

  const current = overrides?.find((o) => o.factor_id === factorId);
  const auditTrail = overrides?.filter((o) => o.factor_id === factorId) ?? [];

  const save = useMutation({
    mutationFn: (payload: {
      override_kind: "value" | "replacement" | "deprecation";
      co2e_total?: number;
      replacement_factor_id?: string;
      reason: string;
      effective_from: string;
      effective_to?: string;
    }) => saveOverride(identity, tenantId, factorId, payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.tenant(tenantId).overrides() }),
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Customer Override Manager</h1>
        <p className="text-sm text-muted-foreground">
          Per-tenant factor overrides. Cross-tenant reads are blocked at the API client.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Scope</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Tenant id</label>
              <Input
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                className="font-mono"
              />
              {tenantId !== identity.tenant_id && !identity.roles.includes("admin") && (
                <p className="mt-1 text-xs text-factor-deprecated-700">
                  Non-admin: only {identity.tenant_id} is accessible.
                </p>
              )}
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Factor id</label>
              <Input
                value={factorId}
                onChange={(e) => setFactorId(e.target.value)}
                className="font-mono"
                placeholder="DEFRA-ELEC-GB-2025-001"
              />
            </div>
            <Button variant="ghost" onClick={() => setFactorId("")}>
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            Existing overrides for <code>{tenantId}</code>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left">
                <tr>
                  <th className="px-3 py-2">factor_id</th>
                  <th className="px-3 py-2">kind</th>
                  <th className="px-3 py-2">value</th>
                  <th className="px-3 py-2">since</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {(overrides ?? []).map((o) => (
                  <tr key={o.factor_id} className="border-t border-border">
                    <td className="px-3 py-2 font-mono text-xs">{o.factor_id}</td>
                    <td className="px-3 py-2">
                      <Badge variant="outline">{o.override_kind}</Badge>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {o.co2e_total ?? o.replacement_factor_id ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {formatDate(o.effective_from)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <Button size="sm" variant="ghost" onClick={() => setFactorId(o.factor_id)}>
                        Edit
                      </Button>
                    </td>
                  </tr>
                ))}
                {(!overrides || overrides.length === 0) && (
                  <tr>
                    <td colSpan={5} className="px-3 py-6 text-center text-xs text-muted-foreground">
                      No overrides for this tenant.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {factorId && (
        <OverrideEditor
          tenantId={tenantId}
          factorId={factorId}
          current={current}
          auditTrail={auditTrail}
          onSave={async (v) => {
            await save.mutateAsync(v);
          }}
        />
      )}
    </div>
  );
}
