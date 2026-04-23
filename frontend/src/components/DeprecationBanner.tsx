/**
 * DeprecationBanner — alert for Wave 2 `DeprecationStatus`.
 *
 * Renders nothing when the status is "active" or missing. Otherwise
 * shows a colored alert with the effective window, reason, replacement
 * factor id, and optional notice URL.
 */
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { DeprecationStatus } from "@greenlang/factors-sdk";
import { normalizeDeprecation } from "../lib/factorsClient";

export interface DeprecationBannerProps {
  /** Accepts string (pre-Wave-2) OR structured (Wave 2+). */
  status?: string | DeprecationStatus | null;
  /** Override alert severity. Defaults based on status. */
  severity?: "info" | "warning" | "error";
}

function severityFor(status: string): "info" | "warning" | "error" {
  const s = status.toLowerCase();
  if (s === "retired" || s === "removed") return "error";
  if (s === "deprecated" || s === "scheduled_for_removal") return "warning";
  return "info";
}

export function DeprecationBanner({ status, severity }: DeprecationBannerProps) {
  const normalized = normalizeDeprecation(status);
  if (!normalized) return null;

  const statusLabel = (normalized.status ?? "deprecated").toString();
  const effectiveSeverity = severity ?? severityFor(statusLabel);

  return (
    <Alert
      severity={effectiveSeverity}
      role="alert"
      data-testid="deprecation-banner"
      sx={{ alignItems: "flex-start" }}
    >
      <AlertTitle sx={{ textTransform: "capitalize" }}>
        {statusLabel.replace(/_/g, " ")}
      </AlertTitle>
      <Stack spacing={0.5}>
        {normalized.reason && (
          <Typography variant="body2">
            <strong>Reason:</strong> {normalized.reason}
          </Typography>
        )}
        {(normalized.effective_from || normalized.effective_to) && (
          <Typography variant="body2">
            <strong>Effective:</strong>{" "}
            {normalized.effective_from ?? "—"} to {normalized.effective_to ?? "open"}
          </Typography>
        )}
        {normalized.replacement_factor_id && (
          <Typography variant="body2">
            <strong>Replacement:</strong>{" "}
            <code>{normalized.replacement_factor_id}</code>
          </Typography>
        )}
        {normalized.notice_url && (
          <Typography variant="body2">
            <Link
              href={normalized.notice_url}
              target="_blank"
              rel="noreferrer"
              aria-label="Deprecation notice"
            >
              Deprecation notice
            </Link>
          </Typography>
        )}
      </Stack>
    </Alert>
  );
}

export default DeprecationBanner;
