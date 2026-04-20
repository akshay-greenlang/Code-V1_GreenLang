/**
 * F7.7 — Impact simulator UI.
 *
 * Wraps the Phase F6 ImpactSimulator. Backing:
 *   POST /api/v1/factors/impact/simulate
 *     { replaced_factor_ids: [...], value_map: {...} }
 */
import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
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
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

interface ImpactedComputation {
  computation_id: string;
  tenant_id: string | null;
  factor_id: string;
  old_value: number | null;
  new_value: number | null;
  delta_abs: number | null;
  delta_pct: number | null;
  evidence_bundle: string | null;
}

interface ImpactReport {
  simulated_at: string;
  affected_factor_ids: string[];
  tenants: string[];
  computation_count: number;
  summary: Record<string, number>;
  computations: ImpactedComputation[];
}

export function FactorsImpactSimulator() {
  const [factorsInput, setFactorsInput] = useState("");
  const [report, setReport] = useState<ImpactReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    const ids = factorsInput
      .split(/[\n,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length === 0) {
      setError("Enter at least one factor_id");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/factors/impact/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ replaced_factor_ids: ids }),
      });
      if (!res.ok) throw new Error(`impact ${res.status}`);
      setReport((await res.json()) as ImpactReport);
    } catch (e) {
      setError((e as Error).message);
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Impact Simulator</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        "What breaks if we replace the UK-2025 road freight factor pack?"
        Traces every ledger entry + evidence bundle that depends on a given
        set of factor IDs.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            fullWidth
            multiline
            minRows={3}
            label="Factor IDs (one per line or comma-separated)"
            placeholder={"EF:UK:road_freight_40t:2025:v1\nEF:UK:road_freight_18t:2025:v1"}
            value={factorsInput}
            onChange={(e) => setFactorsInput(e.target.value)}
          />
          <Button variant="contained" sx={{ mt: 2 }} onClick={() => void run()}>
            Simulate
          </Button>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {report && (
        <>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid xs={12} md={3}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline">Affected computations</Typography>
                  <Typography variant="h3">{report.computation_count}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={3}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline">Affected tenants</Typography>
                  <Typography variant="h3">{report.tenants.length}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={3}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline">Max Δ%</Typography>
                  <Typography variant="h3">
                    {(report.summary.max_pct_delta ?? 0).toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={3}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline">Factors replaced</Typography>
                  <Typography variant="h3">{report.affected_factor_ids.length}</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Computation</TableCell>
                  <TableCell>Tenant</TableCell>
                  <TableCell>Factor</TableCell>
                  <TableCell align="right">Old value</TableCell>
                  <TableCell align="right">New value</TableCell>
                  <TableCell align="right">Δ abs</TableCell>
                  <TableCell align="right">Δ %</TableCell>
                  <TableCell>Evidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {report.computations.map((c) => (
                  <TableRow key={`${c.computation_id}:${c.factor_id}`}>
                    <TableCell><code>{c.computation_id}</code></TableCell>
                    <TableCell>{c.tenant_id ?? "—"}</TableCell>
                    <TableCell><code>{c.factor_id}</code></TableCell>
                    <TableCell align="right">{c.old_value ?? "—"}</TableCell>
                    <TableCell align="right">{c.new_value ?? "—"}</TableCell>
                    <TableCell align="right">{c.delta_abs?.toFixed(2) ?? "—"}</TableCell>
                    <TableCell align="right">{c.delta_pct?.toFixed(2) ?? "—"}</TableCell>
                    <TableCell>
                      {c.evidence_bundle ? (
                        <Chip label={c.evidence_bundle.slice(0, 10) + "…"} size="small" />
                      ) : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
}

export default FactorsImpactSimulator;
