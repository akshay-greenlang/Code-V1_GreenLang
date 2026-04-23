/**
 * Phase 5.3 / Track B-2 — Public catalog status dashboard.
 *
 * Wired to the FY27 spec endpoint `GET /v1/coverage` via factorsClient.
 * Renders three-label counts per family (Certified / Preview /
 * Connector-only) plus optional per-source breakdown.
 *
 * Refresh cadence: 60s (per FY27 launch checklist Track B-2).
 *
 * Public — no auth required. The factorsClient still attaches a bearer
 * token if one is in localStorage, but the endpoint is open.
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
import {
  FactorsApiError,
  getCoverage,
  type CoverageByFamily,
  type CoverageResponse,
  type CoverageTotals,
} from "../lib/api/factorsClient";

interface CoverageMatrixCell extends CoverageTotals {
  family: string;
  jurisdiction: string;
}

/**
 * W4-D: Derive a per-family × per-jurisdiction matrix from the coverage
 * response. The v1.2.0 coverage endpoint SHOULD expose a `by_family_jurisdiction`
 * array; when it doesn't (older server) we fall back to a one-column
 * matrix keyed on "ALL" so the page still renders usefully.
 */
function deriveMatrix(resp: CoverageResponse): CoverageMatrixCell[] {
  const extra = resp as unknown as { by_family_jurisdiction?: CoverageMatrixCell[] };
  if (Array.isArray(extra.by_family_jurisdiction)) return extra.by_family_jurisdiction;
  return (resp.by_family ?? []).map((f) => ({
    ...f,
    jurisdiction: "ALL",
  }));
}

const REFRESH_MS = 60_000;

function ProportionBar({ totals }: { totals: CoverageTotals }) {
  const total = Math.max(1, totals.all);
  const segments = [
    { label: "Certified", value: totals.certified, color: "#2e7d32" },
    { label: "Preview", value: totals.preview, color: "#ed6c02" },
    { label: "Connector-only", value: totals.connector_only, color: "#6c757d" },
    { label: "Deprecated", value: totals.deprecated ?? 0, color: "#c62828" },
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
              sx={{ width: `${pct}%`, backgroundColor: s.color }}
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
  const [summary, setSummary] = useState<CoverageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async (initial: boolean) => {
      if (initial) setLoading(true);
      try {
        const data = await getCoverage();
        if (cancelled) return;
        setSummary(data);
        setError(null);
        setLastRefreshed(new Date());
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      } finally {
        if (!cancelled && initial) setLoading(false);
      }
    };
    void load(true);
    const timer = window.setInterval(() => void load(false), REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
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

  if (loading && !summary) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (error && !summary) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Could not load catalog status: {error}</Alert>
      </Box>
    );
  }

  if (!summary) return null;

  const familyRows: CoverageByFamily[] = summary.by_family ?? [];
  const sourceRows = summary.by_source ?? [];

  return (
    <Box sx={{ p: 3, maxWidth: 1100, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        GreenLang Factors — Catalog Status
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Public coverage snapshot. Three labels separate what's regulator-ready
        (Certified) from what's under review (Preview) and what requires a
        pre-licensed connector (Connector-only). Edition: <code>{summary.edition_id}</code>.
        Generated: {generatedAt}.{" "}
        {lastRefreshed && (
          <span>
            Auto-refresh every 60s, last updated {lastRefreshed.toLocaleTimeString()}.
          </span>
        )}
      </Typography>

      {error && (
        <Alert severity="warning" sx={{ mt: 2 }} role="status">
          Refresh failed ({error}). Showing previously loaded snapshot.
        </Alert>
      )}

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6">Totals</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
            <StatusChip label="Certified" value={summary.totals.certified} />
            <StatusChip label="Preview" value={summary.totals.preview} />
            <StatusChip label="Connector-only" value={summary.totals.connector_only} />
            <StatusChip label="Deprecated" value={summary.totals.deprecated ?? 0} />
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
            By Family
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="Factor counts by family">
              <TableHead>
                <TableRow>
                  <TableCell>Family</TableCell>
                  <TableCell align="right">Certified</TableCell>
                  <TableCell align="right">Preview</TableCell>
                  <TableCell align="right">Connector-only</TableCell>
                  <TableCell align="right">Deprecated</TableCell>
                  <TableCell align="right">Total</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {familyRows.map((row) => (
                  <TableRow key={row.family} hover>
                    <TableCell component="th" scope="row">
                      <code>{row.family}</code>
                    </TableCell>
                    <TableCell align="right">{row.certified.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.preview.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.connector_only.toLocaleString()}</TableCell>
                    <TableCell align="right">{(row.deprecated ?? 0).toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <strong>{row.all.toLocaleString()}</strong>
                    </TableCell>
                  </TableRow>
                ))}
                {familyRows.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No families reported in this edition yet.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {sourceRows.length > 0 && (
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
                  {sourceRows.map((row) => (
                    <TableRow key={row.source_id} hover>
                      <TableCell component="th" scope="row">
                        <code>{row.source_id}</code>
                      </TableCell>
                      <TableCell align="right">{row.certified.toLocaleString()}</TableCell>
                      <TableCell align="right">{row.preview.toLocaleString()}</TableCell>
                      <TableCell align="right">{row.connector_only.toLocaleString()}</TableCell>
                      <TableCell align="right">{(row.deprecated ?? 0).toLocaleString()}</TableCell>
                      <TableCell align="right">
                        <strong>{row.all.toLocaleString()}</strong>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* W4-D: per-family × per-jurisdiction matrix */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Family × Jurisdiction matrix
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="Family by jurisdiction coverage matrix">
              <TableHead>
                <TableRow>
                  <TableCell>Family</TableCell>
                  <TableCell>Jurisdiction</TableCell>
                  <TableCell align="right">Certified</TableCell>
                  <TableCell align="right">Preview</TableCell>
                  <TableCell align="right">Connector-only</TableCell>
                  <TableCell align="right">Total</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {deriveMatrix(summary).map((cell, idx) => (
                  <TableRow key={`${cell.family}-${cell.jurisdiction}-${idx}`} hover>
                    <TableCell><code>{cell.family}</code></TableCell>
                    <TableCell><code>{cell.jurisdiction}</code></TableCell>
                    <TableCell align="right">{cell.certified.toLocaleString()}</TableCell>
                    <TableCell align="right">{cell.preview.toLocaleString()}</TableCell>
                    <TableCell align="right">{cell.connector_only.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <strong>{cell.all.toLocaleString()}</strong>
                    </TableCell>
                  </TableRow>
                ))}
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
