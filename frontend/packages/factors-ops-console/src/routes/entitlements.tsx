import { useState, useEffect } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import { listEntitlements, updateEntitlement } from "@/lib/api";
import type { Identity } from "@/lib/auth";
import {
  type Entitlement,
  type PackAssignment,
  type Tier,
} from "@/types/ops";

export const Route = createFileRoute("/entitlements")({
  component: EntitlementsPage,
});

const ALL_PACKS: PackAssignment[] = [
  "corporate",
  "electricity",
  "freight",
  "eu_policy",
  "land_removals",
  "product_carbon",
  "finance_proxy",
];

function EntitlementsPage() {
  return (
    <AuthGuard requiredAction="entitlement.edit">
      {(identity) => <EntitlementsAdmin identity={identity} />}
    </AuthGuard>
  );
}

function EntitlementsAdmin({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const [tenantId, setTenantId] = useState(identity.tenant_id);
  const { data } = useQuery({
    queryKey: queryKeys.tenant(tenantId).entitlements(),
    queryFn: () => listEntitlements(identity, tenantId),
    enabled: Boolean(tenantId),
  });

  const [draft, setDraft] = useState<Entitlement | null>(null);
  const [reason, setReason] = useState("");

  useEffect(() => {
    if (data) setDraft(data);
  }, [data]);

  const save = useMutation({
    mutationFn: (next: Entitlement & { reason: string }) =>
      updateEntitlement(identity, tenantId, {
        tier: next.tier,
        packs: next.packs,
        rate_limit_rpm: next.rate_limit_rpm,
        preview_access: next.preview_access,
        connector_access: next.connector_access,
        reason: next.reason,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.tenant(tenantId).entitlements() });
      setReason("");
    },
  });

  if (!draft) {
    return (
      <div className="space-y-4">
        <header>
          <h1 className="text-2xl font-semibold">Entitlements Admin</h1>
        </header>
        <Card>
          <CardContent className="py-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Tenant id</label>
              <Input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const togglePack = (p: PackAssignment) => {
    setDraft({
      ...draft,
      packs: draft.packs.includes(p)
        ? draft.packs.filter((x) => x !== p)
        : [...draft.packs, p],
    });
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Entitlements · {tenantId}</h1>
        <p className="text-sm text-muted-foreground">
          Tier + pack assignments per tenant. Every save writes to the audit log.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Tier & packs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div>
            <label className="text-xs text-muted-foreground">Tenant id</label>
            <Input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Tier</label>
            <select
              value={draft.tier}
              onChange={(e) => setDraft({ ...draft, tier: e.target.value as Tier })}
              className="mt-1 h-9 w-full rounded-md border border-border bg-background px-2 text-sm"
            >
              <option value="free">Free</option>
              <option value="starter">Starter</option>
              <option value="pro">Pro</option>
              <option value="enterprise">Enterprise</option>
            </select>
          </div>
          <div>
            <span className="text-xs text-muted-foreground">Packs</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {ALL_PACKS.map((p) => {
                const on = draft.packs.includes(p);
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => togglePack(p)}
                    className={`rounded-md border border-border px-2 py-1 text-xs ${
                      on ? "bg-primary text-primary-foreground" : "bg-background"
                    }`}
                    aria-pressed={on}
                  >
                    {p}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Rate limit (rpm)</label>
              <Input
                type="number"
                value={draft.rate_limit_rpm}
                onChange={(e) =>
                  setDraft({ ...draft, rate_limit_rpm: Number(e.target.value) })
                }
              />
            </div>
            <div className="flex flex-col gap-2 pt-5">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={draft.preview_access}
                  onChange={(e) => setDraft({ ...draft, preview_access: e.target.checked })}
                />
                Preview access
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={draft.connector_access}
                  onChange={(e) => setDraft({ ...draft, connector_access: e.target.checked })}
                />
                Connector access
              </label>
            </div>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Audit reason (required)</label>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={2}
              placeholder="Tier upgraded per SalesOps ticket #12345"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              disabled={reason.trim().length < 10}
              onClick={() => save.mutate({ ...draft, reason })}
            >
              Save
            </Button>
            <Badge variant="muted">writes audit log</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
