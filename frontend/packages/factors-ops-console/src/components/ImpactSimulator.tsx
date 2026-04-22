import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Download, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ImpactSimulationResult } from "@/types/ops";

/**
 * Impact simulator form + preview of affected computations / tenants.
 *
 * Binds to `runImpactSimulation` in api.ts. Results are kept locally — when the
 * user exports, we dump the full JSON blob the backend produced.
 */
const FormSchema = z.object({
  factor_id: z.string().min(1, "Factor id required"),
  mode: z.enum(["listing_only", "value_override", "deprecation"]),
  hypothetical_value: z
    .preprocess(
      (v) => (v === "" || v === null || v === undefined ? undefined : Number(v)),
      z.number().positive().optional()
    ),
  tenant_scope_raw: z.string().optional(),
  reason: z.string().min(10, "Reason must be at least 10 characters").max(500),
});
export type ImpactSimInput = z.infer<typeof FormSchema>;

interface Props {
  onRun: (payload: {
    factor_id: string;
    mode: "listing_only" | "value_override" | "deprecation";
    hypothetical_value?: number;
    tenant_scope?: string[];
    reason: string;
  }) => Promise<ImpactSimulationResult>;
}

export function ImpactSimulator({ onRun }: Props) {
  const [result, setResult] = useState<ImpactSimulationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isValid },
  } = useForm<ImpactSimInput>({
    resolver: zodResolver(FormSchema),
    mode: "onChange",
    defaultValues: { mode: "value_override", tenant_scope_raw: "" },
  });
  const mode = watch("mode");

  const submit = handleSubmit(async (v) => {
    setBusy(true);
    try {
      const tenants = (v.tenant_scope_raw ?? "")
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const r = await onRun({
        factor_id: v.factor_id,
        mode: v.mode,
        hypothetical_value: v.hypothetical_value,
        tenant_scope: tenants.length ? tenants : undefined,
        reason: v.reason,
      });
      setResult(r);
    } finally {
      setBusy(false);
    }
  });

  const exportJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `impact-sim-${result.simulation_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>What-if simulation</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="space-y-3" onSubmit={submit}>
            <div>
              <label className="text-sm font-medium">Factor id</label>
              <Input {...register("factor_id")} className="font-mono" placeholder="DEFRA-RF-GB-2024" />
              {errors.factor_id && (
                <p className="text-xs text-factor-deprecated-700">{errors.factor_id.message}</p>
              )}
            </div>
            <div>
              <label className="text-sm font-medium">Mode</label>
              <select
                {...register("mode")}
                className="mt-1 h-9 w-full rounded-md border border-border bg-background px-2 text-sm"
              >
                <option value="listing_only">listing only</option>
                <option value="value_override">value override</option>
                <option value="deprecation">deprecation</option>
              </select>
            </div>
            {mode === "value_override" && (
              <div>
                <label className="text-sm font-medium">Hypothetical co2e_total</label>
                <Input type="number" step="0.0001" {...register("hypothetical_value")} />
              </div>
            )}
            <div>
              <label className="text-sm font-medium">Tenant scope (comma-separated, empty = all)</label>
              <Input {...register("tenant_scope_raw")} placeholder="acme-corp, globex" />
            </div>
            <div>
              <label className="text-sm font-medium">Audit reason</label>
              <Textarea rows={2} {...register("reason")} />
              {errors.reason && (
                <p className="text-xs text-factor-deprecated-700">{errors.reason.message}</p>
              )}
            </div>
            <Button type="submit" disabled={!isValid || busy}>
              <Play className="h-4 w-4" /> Run simulation
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Affected downstream</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {!result ? (
            <p className="text-muted-foreground">Run a simulation to preview affected computations and tenants.</p>
          ) : (
            <>
              <dl className="grid grid-cols-2 gap-2">
                <Stat label="Affected computations" value={result.summary.affected_computations.toLocaleString()} />
                <Stat label="Affected tenants" value={String(result.summary.affected_tenants)} />
                <Stat label="Avg Δ" value={`${result.summary.avg_delta_pct.toFixed(2)}%`} />
                <Stat label="Max Δ" value={`${result.summary.max_delta_pct.toFixed(2)}%`} />
              </dl>
              {result.suggested_rollback_plan && (
                <p className="rounded-md border border-border bg-muted/30 p-2 text-xs">
                  <strong>Rollback plan:</strong> {result.suggested_rollback_plan}
                </p>
              )}
              <Button variant="outline" size="sm" onClick={exportJson}>
                <Download className="h-3.5 w-3.5" /> Export JSON
              </Button>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono text-base">{value}</dd>
    </div>
  );
}
