import { useQuery } from "@tanstack/react-query";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";
import { getCoverage } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Card, CardContent } from "@/components/ui/card";
import { cn, formatDate } from "@/lib/utils";

interface ThreeLabelDashboardProps {
  /** Optional trend deltas in basis points (provided by future snapshot endpoint). */
  trend?: {
    certified?: number;
    preview?: number;
    connector_only?: number;
  };
}

/**
 * Three counters (Certified / Preview / Connector-only) with last-updated
 * timestamp and a placeholder trend arrow. The counts come from
 * /api/v1/factors/coverage.
 */
export function ThreeLabelDashboard({ trend }: ThreeLabelDashboardProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.coverage(),
    queryFn: getCoverage,
  });

  if (error) {
    return (
      <div
        role="alert"
        className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
      >
        Coverage summary temporarily unavailable.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <CounterCard
          label="Certified"
          value={data?.certified_count ?? null}
          delta={trend?.certified}
          statusClass="status-certified"
          loading={isLoading}
        />
        <CounterCard
          label="Preview"
          value={data?.preview_count ?? null}
          delta={trend?.preview}
          statusClass="status-preview"
          loading={isLoading}
        />
        <CounterCard
          label="Connector-only"
          value={data?.connector_only_count ?? null}
          delta={trend?.connector_only}
          statusClass="status-connector-only"
          loading={isLoading}
        />
      </div>
      {data ? (
        <p className="text-xs text-muted-foreground">
          Edition <span className="font-mono">{data.edition_id}</span> •
          generated {formatDate(data.generated_at)} •{" "}
          <span className="tabular-nums">{data.total.toLocaleString()}</span>{" "}
          factors total
        </p>
      ) : null}
    </div>
  );
}

function CounterCard({
  label,
  value,
  delta,
  statusClass,
  loading,
}: {
  label: string;
  value: number | null;
  delta: number | undefined;
  statusClass: string;
  loading: boolean;
}) {
  return (
    <Card className={cn("border-l-4", statusClass)}>
      <CardContent className="flex flex-col gap-1 p-4">
        <span className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <span className="text-3xl font-semibold tabular-nums">
          {loading || value === null ? "—" : value.toLocaleString()}
        </span>
        <span className="flex items-center gap-1 text-xs text-muted-foreground">
          <TrendArrow delta={delta} />
          {delta === undefined
            ? "trend coming soon"
            : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}% vs prev edition`}
        </span>
      </CardContent>
    </Card>
  );
}

function TrendArrow({ delta }: { delta: number | undefined }) {
  if (delta === undefined) return <Minus className="h-3 w-3 opacity-40" />;
  if (delta > 0) return <ArrowUp className="h-3 w-3 text-emerald-600" />;
  if (delta < 0) return <ArrowDown className="h-3 w-3 text-rose-600" />;
  return <Minus className="h-3 w-3 opacity-40" />;
}
