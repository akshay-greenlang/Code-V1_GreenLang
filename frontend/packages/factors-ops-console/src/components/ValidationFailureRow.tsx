import { AlertOctagon, AlertTriangle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { ValidationFailure } from "@/types/ops";

interface Props {
  failure: ValidationFailure;
  canRemediate: boolean;
  onRemediate: (f: ValidationFailure, resolution: "approve" | "reject" | "fix") => void;
}

const SEVERITY_META: Record<
  ValidationFailure["severity"],
  { label: string; variant: "danger" | "warn" | "muted"; icon: typeof Info }
> = {
  critical: { label: "critical", variant: "danger", icon: AlertOctagon },
  high: { label: "high", variant: "danger", icon: AlertOctagon },
  med: { label: "med", variant: "warn", icon: AlertTriangle },
  low: { label: "low", variant: "muted", icon: Info },
};

export function ValidationFailureRow({ failure, canRemediate, onRemediate }: Props) {
  const meta = SEVERITY_META[failure.severity];
  const Icon = meta.icon;
  return (
    <tr className="border-t border-border hover:bg-muted/30 align-top">
      <td className="px-3 py-2 font-mono text-xs">{failure.id}</td>
      <td className="px-3 py-2 text-xs">
        <Badge variant="outline" className="capitalize">
          {failure.module.replace("_", " ")}
        </Badge>
      </td>
      <td className="px-3 py-2">
        <Badge variant={meta.variant} className="gap-1">
          <Icon className="h-3 w-3" aria-hidden="true" />
          {meta.label}
        </Badge>
      </td>
      <td className="px-3 py-2 font-mono text-xs">
        {failure.factor_id ?? <span className="text-muted-foreground">—</span>}
      </td>
      <td className="px-3 py-2 text-sm">
        <div>{failure.message}</div>
        {failure.remediation_hint && (
          <div className="mt-0.5 text-xs text-muted-foreground">
            hint: {failure.remediation_hint}
          </div>
        )}
      </td>
      <td className="px-3 py-2 text-right">
        <div className="flex items-center justify-end gap-1">
          {canRemediate ? (
            <>
              <Button size="sm" variant="default" onClick={() => onRemediate(failure, "approve")}>
                Approve
              </Button>
              <Button size="sm" variant="outline" onClick={() => onRemediate(failure, "fix")}>
                Fix
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onRemediate(failure, "reject")}>
                Reject
              </Button>
            </>
          ) : (
            <span className="text-xs text-muted-foreground">read-only</span>
          )}
        </div>
      </td>
    </tr>
  );
}
