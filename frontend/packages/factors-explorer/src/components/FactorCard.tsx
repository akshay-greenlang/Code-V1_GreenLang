import { Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import type { FactorRecord } from "@/types/factors";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LicenseBadge } from "@/components/LicenseBadge";
import { QualityMeter } from "@/components/QualityMeter";
import { cn, formatDate } from "@/lib/utils";

interface FactorCardProps {
  factor: FactorRecord;
  highlight?: string;
}

/**
 * List-view card: factor_id, factor_family badge, jurisdiction, source_id,
 * valid range, FQS meter, license badge. Clicking navigates to /factors/:id.
 */
export function FactorCard({ factor, highlight }: FactorCardProps) {
  const statusBadge = statusVariant(factor.factor_status);

  return (
    <Card className="transition-shadow hover:shadow-md">
      <Link
        to="/factors/$factorId"
        params={{ factorId: factor.factor_id }}
        className="block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label={`Open ${factor.factor_id}`}
      >
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs text-muted-foreground">
                  {factor.factor_id}
                </span>
                <Badge variant="outline" className="uppercase">
                  {factor.factor_family}
                </Badge>
                <Badge variant="outline">scope {factor.scope}</Badge>
                <Badge variant={statusBadge.variant}>
                  {statusBadge.label}
                </Badge>
              </div>

              <h3 className="mt-1 truncate text-base font-semibold">
                {factor.fuel_type || factor.factor_id}
                {factor.jurisdiction ? (
                  <span className="text-muted-foreground">
                    {" "}
                    — {factor.jurisdiction}
                  </span>
                ) : null}
              </h3>

              <p
                className={cn(
                  "mt-0.5 text-sm text-muted-foreground",
                  highlight ? "" : "truncate"
                )}
                // highlight from /search endpoint arrives server-sanitized
                dangerouslySetInnerHTML={
                  highlight
                    ? { __html: highlight }
                    : undefined
                }
              >
                {highlight
                  ? undefined
                  : `${factor.provenance.source_org} ${factor.provenance.source_year} v${factor.provenance.source_version || "—"}`}
              </p>

              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                <span className="tabular-nums font-medium text-foreground">
                  {factor.co2e_per_unit.toFixed(4)} kgCO₂e / {factor.unit}
                </span>
                <span>
                  valid {formatDate(factor.valid_from)}
                  {factor.valid_to ? ` → ${formatDate(factor.valid_to)}` : ""}
                </span>
                {factor.source_id ? (
                  <span className="font-mono">{factor.source_id}</span>
                ) : null}
              </div>

              <div className="mt-2 flex flex-wrap items-center gap-2">
                <QualityMeter fqs={factor.quality} compact />
                <LicenseBadge
                  licenseClass={factor.license_info.class}
                  redistributionAllowed={
                    factor.license_info.redistribution_allowed
                  }
                  commercialUseAllowed={
                    factor.license_info.commercial_use_allowed
                  }
                  attributionRequired={factor.license_info.attribution_required}
                  compact
                />
              </div>
            </div>

            <ArrowRight
              className="mt-2 h-5 w-5 shrink-0 opacity-40 transition group-hover:opacity-100"
              aria-hidden="true"
            />
          </div>
        </CardContent>
      </Link>
    </Card>
  );
}

function statusVariant(status: FactorRecord["factor_status"]): {
  variant: "success" | "warning" | "info" | "destructive";
  label: string;
} {
  switch (status) {
    case "certified":
      return { variant: "success", label: "certified" };
    case "preview":
      return { variant: "warning", label: "preview" };
    case "connector_only":
      return { variant: "info", label: "connector-only" };
    case "deprecated":
      return { variant: "destructive", label: "deprecated" };
  }
}
