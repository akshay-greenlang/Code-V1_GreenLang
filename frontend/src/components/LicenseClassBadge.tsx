/**
 * LicenseClassBadge — colored chip for the v1.2.0 `LicensingEnvelope.license_class`.
 *
 * Four canonical classes:
 *   certified       → green   (regulator-ready)
 *   preview         → amber   (usable with disclosure)
 *   connector_only  → gray    (access requires upstream license)
 *   redistributable → blue    (distributable to sub-tenants)
 */
import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import { licenseClassLabel, licenseClassColor } from "../lib/factorsClient";

const TOOLTIP: Record<string, string> = {
  certified:
    "Certified — signed off by curation + license review, safe for regulatory filings.",
  preview:
    "Preview — usable with explicit disclosure; pending methodology sign-off.",
  connector_only:
    "Connector-only — rights-restricted. Access requires a pre-licensed connector for the caller's tenant.",
  redistributable:
    "Redistributable — license allows downstream distribution (OEM white-label, sub-tenants).",
};

const BG_COLOR: Record<string, { bg: string; fg: string }> = {
  certified: { bg: "#2e7d32", fg: "#fff" },
  preview: { bg: "#ed6c02", fg: "#fff" },
  connector_only: { bg: "#6c757d", fg: "#fff" },
  redistributable: { bg: "#0277bd", fg: "#fff" },
};

export interface LicenseClassBadgeProps {
  cls?: string | null;
  /** Hide the hover tooltip (for use inside dense tables). */
  noTooltip?: boolean;
  size?: "small" | "medium";
}

export function LicenseClassBadge({ cls, noTooltip = false, size = "small" }: LicenseClassBadgeProps) {
  const normalized = (cls ?? "").toLowerCase();
  const palette = BG_COLOR[normalized];
  const chipProps: React.ComponentProps<typeof Chip> = {
    label: licenseClassLabel(cls),
    size,
    "aria-label": `License class ${licenseClassLabel(cls)}`,
  };
  if (palette) {
    chipProps.sx = { bgcolor: palette.bg, color: palette.fg, fontWeight: 600 };
  } else {
    chipProps.color = licenseClassColor(cls);
    chipProps.variant = "outlined";
  }
  const chip = <Chip {...chipProps} data-testid="license-class-badge" />;
  if (noTooltip) return chip;
  const tip = TOOLTIP[normalized];
  return tip ? (
    <Tooltip title={tip} arrow>
      <span>{chip}</span>
    </Tooltip>
  ) : (
    chip
  );
}

export default LicenseClassBadge;
