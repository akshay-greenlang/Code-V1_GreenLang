import Chip from "@mui/material/Chip";

export type ShellStatusChip = "PASS" | "WARN" | "FAIL" | "NEUTRAL";

export function apiStatusChipFromResponse(s?: string): ShellStatusChip | undefined {
  if (s === "PASS" || s === "WARN" || s === "FAIL") return s;
  return undefined;
}

export function mapRunStateToStatusChip(runState: string | undefined): ShellStatusChip {
  const rs = (runState || "").toLowerCase();
  if (rs === "failed" || rs === "blocked") return "FAIL";
  if (rs === "partial_success") return "WARN";
  if (rs === "completed") return "PASS";
  return "NEUTRAL";
}

const chipColor: Record<ShellStatusChip, "success" | "warning" | "error" | "default"> = {
  PASS: "success",
  WARN: "warning",
  FAIL: "error",
  NEUTRAL: "default"
};

export interface RunStatusChipProps {
  runState?: string;
  /** When set, overrides mapping from runState */
  chip?: ShellStatusChip;
  size?: "small" | "medium";
}

export function RunStatusChip({ runState, chip, size = "small" }: RunStatusChipProps) {
  const kind = chip ?? mapRunStateToStatusChip(runState);
  const label = kind === "NEUTRAL" ? "—" : kind;
  return (
    <Chip
      component="span"
      size={size}
      label={label}
      color={chipColor[kind]}
      variant={kind === "NEUTRAL" ? "outlined" : "filled"}
      sx={{ fontWeight: 600, verticalAlign: "middle" }}
    />
  );
}
