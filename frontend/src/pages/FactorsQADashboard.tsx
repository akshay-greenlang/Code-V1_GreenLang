/**
 * Track B-2 / F7.3 — Public FQS dashboard.
 *
 * Wired to `GET /v1/quality/fqs`: composite Factor Quality Score
 * distribution per family + drill-down to component scores
 * (temporal / geographic / technology / verification / completeness).
 *
 * Refresh cadence: 60s.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  getFqs,
  type FqsByFamily,
  type FqsResponse,
} from "../lib/api/factorsClient";

const REFRESH_MS = 60_000;

function fqsColor(score: number): "success" | "warning" | "error" {
  if (score >= 75) return "success";
  if (score >= 50) return "warning";
  return "error";
}

function DistributionBar({ buckets }: { buckets: Record<string, number> }) {
  const entries = Object.entries(buckets).sort(([a], [b]) => a.localeCompare(b));
  const total = Math.max(1, entries.reduce((sum, [, n]) => sum + n, 0));
  const palette = ["#c62828", "#ed6c02", "#f9a825", "#9ccc65", "#2e7d32"];
  return (
    <Box sx={{ width: "100%" }} aria-label="FQS distribution">
      <Box
        role="img"
        aria-label="FQS distribution bar"
        sx={{
          display: "flex",
          height: 12,
          width: "100%",
          borderRadius: 0.5,
          overflow: "hidden",
          border: "1px solid",
          borderColor: "divider",
        }}
      >
        {entries.map(([bucket, n], idx) => {
          const pct = (n / total) * 100;
          if (pct === 0) return null;
          return (
            <Box
              key={bucket}
              title={`${bucket}: ${n} (${pct.toFixed(1)}%)`}
              sx={{ width: `${pct}%`, backgroundColor: palette[idx % palette.length] }}
            />
          );
        })}
      </Box>
      <Stack direction="row" spacing={1} sx={{ mt: 0.5, flexWrap: "wrap" }}>
        {entries.map(([bucket, n], idx) => (
          <Chip
            key={bucket}
            size="small"
            variant="outlined"
            label={`${bucket}: ${n}`}
            sx={{ borderColor: palette[idx % palette.length] }}
          />
        ))}
      </Stack>
    </Box>
  );
}

function ComponentScoreRow({ family }: { family: FqsByFamily }) {
  const c = family.components_mean;
  const cells: Array<{ label: string; value: number }> = [
    { label: "Temporal", value: c.temporal },
    { label: "Geographic", value: c.geographic },
    { label: "Technology", value: c.technology },
    { label: "Verification", value: c.verification },
    { label: "Completeness", value: c.completeness },
  ];
  return (
    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
      {cells.map((cell) => (
        <Chip
          key={cell.label}
          size="small"
          variant="outlined"
          color={fqsColor(cell.value)}
          label={`${cell.label}: ${cell.value.toFixed(1)}`}
        />
      ))}
    </Stack>
  );
}

export function FactorsQADashboard() {
  const [data, setData] = useState<FqsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async (initial: boolean) => {
      if (initial) setLoading(true);
      try {
        const res = await getFqs();
        if (cancelled) return;
        setData(res);
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

  const totals = useMemo(() => {
    if (!data) return null;
    const families = data.by_family ?? [];
    const count = families.reduce((s, f) => s + f.count, 0);
    const meanFqs =
      count === 0 ? 0 : families.reduce((s, f) => s + f.mean_fqs * f.count, 0) / count;
    return { count, meanFqs };
  }, [data]);

  if (loading && !data) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (error && !data) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Could not load FQS dashboard: {error}</Alert>
      </Box>
    );
  }

  if (!data) return null;

  const selected = selectedFamily
    ? (data.by_family ?? []).find((f) => f.family === selectedFamily)
    : null;

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Factor Quality Score (FQS) Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Composite FQS distribution per family. Click a row to drill into
        temporal / geographic / technology / verification / completeness
        component scores. Edition: <code>{data.edition_id}</code>.{" "}
        {lastRefreshed && (
          <span>Auto-refresh every 60s, last updated {lastRefreshed.toLocaleTimeString()}.</span>
        )}
      </Typography>

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }} role="status">
          Refresh failed ({error}). Showing previously loaded snapshot.
        </Alert>
      )}

      {totals && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid xs={12} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  Factors scored
                </Typography>
                <Typography variant="h3">{totals.count.toLocaleString()}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid xs={12} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  Weighted mean FQS
                </Typography>
                <Typography variant="h3" color={`${fqsColor(totals.meanFqs)}.main`}>
                  {totals.meanFqs.toFixed(1)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid xs={12} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  Families covered
                </Typography>
                <Typography variant="h3">{(data.by_family ?? []).length}</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            FQS by Family
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="FQS by family">
              <TableHead>
                <TableRow>
                  <TableCell>Family</TableCell>
                  <TableCell align="right">Count</TableCell>
                  <TableCell align="right">Mean FQS</TableCell>
                  <TableCell align="right">Median FQS</TableCell>
                  <TableCell>Distribution</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {(data.by_family ?? []).map((family) => (
                  <TableRow
                    key={family.family}
                    hover
                    onClick={() =>
                      setSelectedFamily((cur) =>
                        cur === family.family ? null : family.family,
                      )
                    }
                    sx={{
                      cursor: "pointer",
                      backgroundColor: family.family === selectedFamily ? "action.selected" : undefined,
                    }}
                  >
                    <TableCell><code>{family.family}</code></TableCell>
                    <TableCell align="right">{family.count.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <Chip
                        label={family.mean_fqs.toFixed(1)}
                        color={fqsColor(family.mean_fqs)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="right">{family.median_fqs.toFixed(1)}</TableCell>
                    <TableCell>
                      <DistributionBar buckets={family.distribution} />
                    </TableCell>
                  </TableRow>
                ))}
                {(data.by_family ?? []).length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No FQS data available yet.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {selected && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Component scores — <code>{selected.family}</code>
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Average across {selected.count.toLocaleString()} factors.
            </Typography>
            <ComponentScoreRow family={selected} />
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default FactorsQADashboard;
