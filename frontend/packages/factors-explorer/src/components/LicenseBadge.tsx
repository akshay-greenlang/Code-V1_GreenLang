import { Lock, Globe, Shield, Briefcase } from "lucide-react";
import type { LicenseClass } from "@/types/factors";
import { cn } from "@/lib/utils";

interface LicenseBadgeProps {
  licenseClass: LicenseClass;
  redistributionAllowed?: boolean;
  commercialUseAllowed?: boolean;
  attributionRequired?: boolean;
  compact?: boolean;
}

/**
 * License class badge.
 * open=green, restricted=amber, licensed=indigo, customer_private=slate.
 * customer_private + licensed get a lock icon.
 */
export function LicenseBadge({
  licenseClass,
  redistributionAllowed,
  commercialUseAllowed,
  attributionRequired,
  compact = false,
}: LicenseBadgeProps) {
  const normalized = normalizeLicenseClass(licenseClass);
  const palette = getPalette(normalized);
  const Icon = getIcon(normalized);

  const label = labelFor(licenseClass);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        palette
      )}
      aria-label={`License class: ${label}`}
      title={
        compact
          ? label
          : [
              label,
              redistributionAllowed ? "redistribution ok" : "no redistribution",
              commercialUseAllowed ? "commercial ok" : "non-commercial",
              attributionRequired ? "attribution required" : "no attribution",
            ].join(" • ")
      }
    >
      <Icon className="h-3 w-3" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

function normalizeLicenseClass(
  licenseClass: LicenseClass
): "open" | "restricted" | "licensed" | "customer_private" {
  switch (licenseClass) {
    case "public":
    case "open":
    case "open_cc":
      return "open";
    case "restricted":
    case "proprietary":
      return "restricted";
    case "commercial":
    case "licensed":
      return "licensed";
    case "customer_private":
    case "connector_only":
      return "customer_private";
  }
}

function getPalette(
  kind: "open" | "restricted" | "licensed" | "customer_private"
): string {
  switch (kind) {
    case "open":
      return "bg-emerald-50 text-emerald-700 border-emerald-500";
    case "restricted":
      return "bg-amber-50 text-amber-700 border-amber-500";
    case "licensed":
      return "bg-indigo-50 text-indigo-700 border-indigo-500";
    case "customer_private":
      return "bg-slate-100 text-slate-700 border-slate-400";
  }
}

function getIcon(kind: "open" | "restricted" | "licensed" | "customer_private") {
  switch (kind) {
    case "open":
      return Globe;
    case "restricted":
      return Shield;
    case "licensed":
      return Briefcase;
    case "customer_private":
      return Lock;
  }
}

function labelFor(cls: LicenseClass): string {
  const map: Record<LicenseClass, string> = {
    public: "Public",
    open: "Open",
    open_cc: "Open (CC)",
    commercial: "Commercial",
    restricted: "Restricted",
    licensed: "Licensed",
    proprietary: "Proprietary",
    customer_private: "Private",
    connector_only: "Connector-only",
  };
  return map[cls];
}
