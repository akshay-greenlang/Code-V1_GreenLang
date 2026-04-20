/**
 * Phase 5.3 — Public three-label catalog status dashboard.
 *
 * Shows factor counts by coverage label (Certified / Preview /
 * Connector-only / Deprecated) plus a per-source breakdown. Public:
 * no auth required. Backed by `GET /api/v1/factors/status/summary`.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";

export interface StatusTotals {
  certified: number;
  preview: number;
  connector_only: number;
  deprecated: number;
  all: number;
}

export interface StatusBySource extends StatusTotals {
  source_id: string;
}

export interface StatusSummary {
  edition_id: string;
  totals: StatusTotals;
  by_source: StatusBySource[];
  generated_at: string;
}

/** Proportion bar: a small horizontal stacked bar without charting deps. */
function ProportionBar({ totals }: { totals: StatusTotals }) {
  const total = Math.max(1, totals.all);
  const segments = [
    { label: "Certified", value: totals.certified, color: "#2e7d32" },
    { label: "Preview", value: totals.preview, color: "#ed6c02" },
    { label: "Connector-only", value: totals.connector_only, color: "#6c757d" },
    { label: "Deprecated", value: totals.deprecated, color: "#c62828" },
  ];
  return (
    <Box sx={{ width: "100%" }}>
      <Box
        role="img"
        aria-label="Factor coverage proportion bar"
        sx={{
          display: "flex",
          height: 24,
          width: "100%",
          borderRadius: 1,
          overflow: "hidden",
          border: "1px solid",
          borderColor: "divider",
        }}
      >
        {segments.map((s) => {
          const pct = (s.value / total) * 100;
          if (pct === 0) return null;
          return (
            <Box
              key={s.label}
              title={`${s.label}: ${s.value} (${pct.toFixed(1)}%)`}
              sx={{
                width: `${pct}%`,
                backgroundColor: s.color,
              }}
            />
          );
        })}
      </Box>
      <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
        {segments.map((s) => (
          <Stack key={s.label} direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 10, height: 10, backgroundColor: s.color, borderRadius: 0.5 }} />
            <Typography variant="caption" color="text.secondary">
              {s.label}: <strong>{s.value.toLocaleString()}</strong>
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Box>
  );
}

function StatusChip({ label, value }: { label: string; value: number }) {
  const colorByLabel: Record<string, "success" | "warning" | "default" | "error"> = {
    Certified: "success",
    Preview: "warning",
    "Connector-only": "default",
    Deprecated: "error",
  };
  return (
    <Chip
      label={`${label}: ${value.toLocaleString()}`}
      color={colorByLabel[label] ?? "default"}
      variant="outlined"
    />
  );
}

export function FactorsCatalogStatus() {
  const [summary, setSummary] = useState<StatusSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    // Public endpoint — no auth headers needed.
    fetch("/api/v1/factors/status/summary")
      .then(async (res) => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
      })
      .then((data: StatusSummary) => {
        if (!cancelled) {
          setSummary(data);
          setError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const generatedAt = useMemo(() => {
    if (!summary) return "";
    try {
      return new Date(summary.generated_at).toLocaleString();
    } catch {
      return summary.generated_at;
    }
  }, [summary]);

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Could not load catalog status: {error}</Alert>
      </Box>
    );
  }

  if (!summary) return null;

  return (
    <Box sx={{ p: 3, maxWidth: 1000, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        GreenLang Factors — Catalog Status
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Public coverage snapshot. Three labels separate what's regulator-ready
        (Certified) from what's under review (Preview) and what requires a
        pre-licensed connector (Connector-only). Edition: <code>{summary.edition_id}</code>.
        Generated: {generatedAt}.
      </Typography>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6">Totals</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
            <StatusChip label="Certified" value={summary.totals.certified} />
            <StatusChip label="Preview" value={summary.totals.preview} />
            <StatusChip label="Connector-only" value={summary.totals.connector_only} />
            <StatusChip label="Deprecated" value={summary.totals.deprecated} />
            <Chip label={`Total: ${summary.totals.all.toLocaleString()}`} />
          </Stack>
          <Box sx={{ mt: 3 }}>
            <ProportionBar totals={summary.totals} />
          </Box>
        </CardContent>
      </Card>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            By Source
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="Factor counts by source">
              <TableHead>
                <TableRow>
                  <TableCell>Source</TableCell>
                  <TableCell align="right">Certified</TableCell>
                  <TableCell align="right">Preview</TableCell>
                  <TableCell align="right">Connector-only</TableCell>
                  <TableCell align="right">Deprecated</TableCell>
                  <TableCell align="right">Total</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {summary.by_source.map((row) => (
                  <TableRow key={row.source_id} hover>
                    <TableCell component="th" scope="row">
                      <code>{row.source_id}</code>
                    </TableCell>
                    <TableCell align="right">{row.certified.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.preview.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.connector_only.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.deprecated.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <strong>{row.all.toLocaleString()}</strong>
                    </TableCell>
                  </TableRow>
                ))}
                {summary.by_source.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No factors in this edition.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>What these labels mean:</strong>{" "}
          <strong>Certified</strong> — signed off by curation + license review,
          safe for regulatory filings.{" "}
          <strong>Preview</strong> — usable with explicit disclosure, pending
          signoff.{" "}
          <strong>Connector-only</strong> — rights-restricted, accessible only
          through pre-licensed customer connectors.{" "}
          <strong>Deprecated</strong> — superseded by a newer edition; kept for
          reproducibility.
        </Typography>
      </Alert>
    </Box>
  );
}

export default FactorsCatalogStatus;
