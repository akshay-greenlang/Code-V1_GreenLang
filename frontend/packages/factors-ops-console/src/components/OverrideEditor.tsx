import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDateTime } from "@/lib/utils";
import type { TenantOverlayEntry } from "@/types/ops";

/**
 * Per-tenant factor override editor. The tenant_id is enforced at the page
 * level (URL path) AND at the API client (assertTenantAccess). This component
 * never accepts a cross-tenant payload.
 */
const OverrideFormSchema = z
  .object({
    override_kind: z.enum(["value", "replacement", "deprecation"]),
    co2e_total: z
      .preprocess(
        (v) => (v === "" || v === null || v === undefined ? undefined : Number(v)),
        z.number().positive().optional()
      ),
    replacement_factor_id: z.string().optional(),
    reason: z.string().min(10, "Reason must be at least 10 characters").max(500),
    effective_from: z.string().min(1, "Effective from date required"),
    effective_to: z.string().optional(),
  })
  .refine(
    (v) => v.override_kind !== "value" || typeof v.co2e_total === "number",
    { path: ["co2e_total"], message: "Value override requires co2e_total." }
  )
  .refine(
    (v) => v.override_kind !== "replacement" || Boolean(v.replacement_factor_id),
    { path: ["replacement_factor_id"], message: "Replacement requires a factor id." }
  );

export type OverrideFormInput = z.infer<typeof OverrideFormSchema>;

interface Props {
  tenantId: string;
  factorId: string;
  current?: TenantOverlayEntry;
  auditTrail?: TenantOverlayEntry[];
  onSave: (payload: OverrideFormInput) => Promise<void>;
}

export function OverrideEditor({ tenantId, factorId, current, auditTrail, onSave }: Props) {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting, isValid },
  } = useForm<OverrideFormInput>({
    resolver: zodResolver(OverrideFormSchema),
    mode: "onChange",
    defaultValues: {
      override_kind: current?.override_kind ?? "value",
      co2e_total: current?.co2e_total ?? undefined,
      replacement_factor_id: current?.replacement_factor_id ?? "",
      reason: "",
      effective_from: current?.effective_from?.slice(0, 10) ?? "",
      effective_to: current?.effective_to?.slice(0, 10) ?? "",
    },
  });
  const kind = watch("override_kind");

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2fr_1fr]">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>
              Tenant <code className="font-mono">{tenantId}</code> · factor{" "}
              <code className="font-mono">{factorId}</code>
            </span>
            {current && <Badge variant="outline">existing</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form
            className="space-y-3"
            onSubmit={handleSubmit(async (v) => {
              await onSave(v);
            })}
          >
            <div>
              <label className="text-sm font-medium">Override kind</label>
              <select
                {...register("override_kind")}
                className="mt-1 h-9 w-full rounded-md border border-border bg-background px-2 text-sm"
              >
                <option value="value">value override (co2e_total)</option>
                <option value="replacement">replacement factor</option>
                <option value="deprecation">deprecation</option>
              </select>
            </div>

            {kind === "value" && (
              <div>
                <label className="text-sm font-medium">co2e_total</label>
                <Input type="number" step="0.0001" {...register("co2e_total")} />
                {errors.co2e_total && (
                  <p className="text-xs text-factor-deprecated-700">
                    {errors.co2e_total.message}
                  </p>
                )}
              </div>
            )}
            {kind === "replacement" && (
              <div>
                <label className="text-sm font-medium">Replacement factor id</label>
                <Input {...register("replacement_factor_id")} className="font-mono" />
                {errors.replacement_factor_id && (
                  <p className="text-xs text-factor-deprecated-700">
                    {errors.replacement_factor_id.message}
                  </p>
                )}
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium">Effective from</label>
                <Input type="date" {...register("effective_from")} />
              </div>
              <div>
                <label className="text-sm font-medium">Effective to</label>
                <Input type="date" {...register("effective_to")} />
              </div>
            </div>

            <div>
              <label className="text-sm font-medium">Audit reason (required)</label>
              <Textarea rows={3} {...register("reason")} placeholder="e.g. primary supplier data replaces grid average" />
              {errors.reason && (
                <p className="text-xs text-factor-deprecated-700">{errors.reason.message}</p>
              )}
            </div>

            <Button type="submit" disabled={!isValid || isSubmitting}>
              Save override
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Audit trail</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          {(auditTrail ?? []).slice(0, 10).map((ev, idx) => (
            <div key={idx} className="border-b border-border pb-1 last:border-0">
              <div className="font-mono text-muted-foreground">
                {formatDateTime(ev.created_at)}
              </div>
              <div>
                <strong>{ev.created_by}</strong> {ev.override_kind} override
              </div>
              <div className="text-muted-foreground">{ev.reason}</div>
            </div>
          ))}
          {(!auditTrail || auditTrail.length === 0) && (
            <div className="text-muted-foreground">No audit events yet.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
