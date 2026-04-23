/**
 * FQSGauge — reusable 0-100 Factor Quality Score gauge.
 *
 * Color bands (matches `FACTORS-UI-SPEC`):
 *   0-40   → red    (low quality, escalate)
 *   40-70  → yellow (acceptable, document caveats)
 *   70-100 → green  (regulator-ready)
 *
 * Rendered as a horizontal bar + numeric value + colored label chip.
 * Uses CSS only (no Plotly) so it stays cheap to use in tables.
 */
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

export type FQSBand = "low" | "medium" | "high" | "unknown";

export function fqsBand(score: number | null | undefined): FQSBand {
  if (score === null || score === undefined || Number.isNaN(score)) return "unknown";
  if (score < 40) return "low";
  if (score < 70) return "medium";
  return "high";
}

const BAND_COLOR: Record<FQSBand, string> = {
  low: "#c62828", // red
  medium: "#ed6c02", // yellow/orange
  high: "#2e7d32", // green
  unknown: "#9e9e9e", // gray
};

const BAND_LABEL: Record<FQSBand, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  unknown: "Unknown",
};

export interface FQSGaugeProps {
  /** Composite FQS score (0-100). `null`/`undefined` renders as unknown. */
  score: number | null | undefined;
  /** Inline compact layout (chip + number only). */
  compact?: boolean;
  /** Optional label override (defaults to "FQS"). */
  label?: string;
  /** Override aria-label for screen readers. */
  ariaLabel?: string;
}

export function FQSGauge({ score, compact = false, label = "FQS", ariaLabel }: FQSGaugeProps) {
  const band = fqsBand(score);
  const color = BAND_COLOR[band];
  const pct = typeof score === "number" && !Number.isNaN(score) ? Math.max(0, Math.min(100, score)) : 0;
  const display = typeof score === "number" && !Number.isNaN(score) ? score.toFixed(1) : "—";
  const aria = ariaLabel ?? `Factor quality score ${display} out of 100, band ${BAND_LABEL[band]}`;

  if (compact) {
    return (
      <Chip
        size="small"
        label={`${label} ${display}`}
        sx={{ bgcolor: color, color: "#fff", fontWeight: 600 }}
        aria-label={aria}
        role="img"
      />
    );
  }

  return (
    <Box role="img" aria-label={aria} sx={{ minWidth: 180 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="subtitle2" sx={{ color, fontWeight: 700 }}>
          {display}
        </Typography>
        <Chip
          size="small"
          label={BAND_LABEL[band]}
          sx={{ bgcolor: color, color: "#fff", height: 18 }}
        />
      </Stack>
      <Box
        sx={{
          position: "relative",
          height: 8,
          width: "100%",
          borderRadius: 1,
          bgcolor: "divider",
          overflow: "hidden",
        }}
      >
        {/* Band markers at 40 / 70. */}
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: "40%",
            height: "100%",
            width: 1,
            bgcolor: "rgba(0,0,0,0.25)",
          }}
          aria-hidden
        />
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: "70%",
            height: "100%",
            width: 1,
            bgcolor: "rgba(0,0,0,0.25)",
          }}
          aria-hidden
        />
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: `${pct}%`,
            bgcolor: color,
            transition: "width 200ms ease",
          }}
        />
      </Box>
    </Box>
  );
}

export default FQSGauge;
