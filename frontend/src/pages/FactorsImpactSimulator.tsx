/**
 * F7.7 / Track B-5 / Track C-4 — Impact simulator (marquee operator page).
 *
 * Wired to:
 *   POST /v1/admin/impact-simulate
 *     Body: { from_factor_id, to_factor_id }
 *           OR { from_factor_id, to_factor_payload }
 *           OR { replaced_factor_ids: [...] } (legacy multi-replace)
 *   POST /v1/admin/queue
 *     Body: { factor_id, proposed_status, rationale, evidence }
 *
 * The launch gate explicitly tests the propose → simulate → approve flow,
 * so this page surfaces:
 *   - mode toggle (factor-id pair / proposed-payload / bulk replace)
 *   - KPI tiles: # affected calculations, # affected customers,
 *     # affected inventories, mean Δ%, max Δ%
 *   - distribution histogram (delta-percent buckets)
 *   - "Promote to Preview" button that POSTs the full simulation as
 *     evidence on a new queue item — the methodology lead can then
 *     approve from the Approval Queue page.
 */
import { useMemo, useState, type ReactNode } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
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
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  composeImpactRequest,
  proposeQueueItem,
  proposeQueueItemFromSimulation,
  simulateImpact,
  type ImpactReport,
} from "../lib/api/factorsClient";

type Mode = "single" | "payload" | "bulk";

function buildHistogramBuckets(report: ImpactReport): Array<[string, number]> {
  if (report.distribution && Object.keys(report.distribution).length > 0) {
    return Object.entries(report.distribution).sort(([a], [b]) => a.localeCompare(b));
  }
  // Derive from per-computation deltas if the API didn't summarize.
  const deltas = report.computations
    .map((c) => c.delta_pct)
    .filter((d): d is number => typeof d === "number" && Number.isFinite(d));
  if (deltas.length === 0) return [];
  const buckets: Record<string, number> = {
    "<-50%": 0,
    "-50 to -10%": 0,
    "-10 to -1%": 0,
    "-1 to +1%": 0,
    "+1 to +10%": 0,
    "+10 to +50%": 0,
    ">+50%": 0,
  };
  for (const d of deltas) {
    if (d < -50) buckets["<-50%"]++;
    else if (d < -10) buckets["-50 to -10%"]++;
    else if (d < -1) buckets["-10 to -1%"]++;
    else if (d <= 1) buckets["-1 to +1%"]++;
    else if (d <= 10) buckets["+1 to +10%"]++;
    else if (d <= 50) buckets["+10 to +50%"]++;
    else buckets[">+50%"]++;
  }
  return Object.entries(buckets);
}

function Histogram({ report }: { report: ImpactReport }) {
  const buckets = useMemo(() => buildHistogramBuckets(report), [report]);
  const max = Math.max(1, ...buckets.map(([, n]) => n));
  if (buckets.length === 0) {
    return (
      <Typography variant="caption" color="text.secondary">
        Not enough data for a distribution.
      </Typography>
    );
  }
  return (
    <Box role="img" aria-label="Delta-percent histogram">
      <Stack spacing={0.5}>
        {buckets.map(([label, n]) => (
          <Stack key={label} direction="row" spacing={1} alignItems="center">
            <Typography variant="caption" sx={{ width: 110, fontFamily: "monospace" }}>
              {label}
            </Typography>
            <Box
              sx={{
                height: 14,
                width: `${(n / max) * 100}%`,
                minWidth: n > 0 ? 4 : 0,
                bgcolor: "primary.main",
                borderRadius: 0.5,
              }}
            />
            <Typography variant="caption" color="text.secondary">
              {n.toLocaleString()}
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Box>
  );
}

function StatCard({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="overline" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="h3" sx={{ color }}>
          {value}
        </Typography>
      </CardContent>
    </Card>
  );
}

export function FactorsImpactSimulator() {
  const [mode, setMode] = useState<Mode>("single");
  const [fromFactorId, setFromFactorId] = useState("");
  const [toFactorId, setToFactorId] = useState("");
  const [toFactorPayloadJson, setToFactorPayloadJson] = useState("");
  const [bulkIds, setBulkIds] = useState("");

  const [report, setReport] = useState<ImpactReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [promoteOpen, setPromoteOpen] = useState(false);
  const [promoteRationale, setPromoteRationale] = useState("");
  const [promoting, setPromoting] = useState(false);
  const [promotedReviewId, setPromotedReviewId] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    setPromotedReviewId(null);
    try {
      const req = composeImpactRequest({
        fromFactorId: mode === "bulk" ? undefined : fromFactorId,
        toFactorId: mode === "single" ? toFactorId : undefined,
        toFactorPayloadJson: mode === "payload" ? toFactorPayloadJson : undefined,
        bulkIds: mode === "bulk" ? bulkIds : undefined,
      });
      const r = await simulateImpact(req);
      setReport(r);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  const promote = async () => {
    if (!report) return;
    setPromoting(true);
    setError(null);
    try {
      const payload = proposeQueueItemFromSimulation({
        fromFactorId: mode === "bulk" ? undefined : fromFactorId.trim() || undefined,
        toFactorId: mode === "single" ? toFactorId.trim() || undefined : undefined,
        rationale: promoteRationale.trim() || "Promote-to-preview from impact simulation",
        report,
      });
      const { review_id } = await proposeQueueItem(payload);
      setPromotedReviewId(review_id);
      setPromoteOpen(false);
      setPromoteRationale("");
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setPromoting(false);
    }
  };

  const customerCount = report?.customer_count ?? report?.tenants.length ?? 0;
  const inventoryCount = report?.inventory_count ?? 0;
  const meanDelta = report?.summary.mean_pct_delta;
  const maxDelta = report?.summary.max_pct_delta;

  return (
    <Box sx={{ p: 3, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Impact Simulator
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        "What breaks if we replace this factor?" Traces every calculation, customer, and
        inventory that depends on the original factor and shows the magnitude of each delta.
        Use the <strong>Promote to Preview</strong> button to attach the full simulation as
        evidence on a new queue item.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            <ToggleButtonGroup
              value={mode}
              exclusive
              onChange={(_event, v) => {
                if (v) setMode(v as Mode);
              }}
              size="small"
            >
              <ToggleButton value="single">Replace by ID</ToggleButton>
              <ToggleButton value="payload">Proposed payload</ToggleButton>
              <ToggleButton value="bulk">Bulk replace (legacy)</ToggleButton>
            </ToggleButtonGroup>

            {mode === "single" && (
              <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                <TextField
                  fullWidth
                  required
                  label="From factor ID (the one being replaced)"
                  placeholder="EF:UK:road_freight_40t:2025:v1"
                  value={fromFactorId}
                  onChange={(e) => setFromFactorId(e.target.value)}
                />
                <TextField
                  fullWidth
                  required
                  label="To factor ID (the replacement)"
                  placeholder="EF:UK:road_freight_40t:2026:v1"
                  value={toFactorId}
                  onChange={(e) => setToFactorId(e.target.value)}
                />
              </Stack>
            )}

            {mode === "payload" && (
              <Stack spacing={2}>
                <TextField
                  fullWidth
                  required
                  label="From factor ID (the one being replaced)"
                  placeholder="EF:UK:road_freight_40t:2025:v1"
                  value={fromFactorId}
                  onChange={(e) => setFromFactorId(e.target.value)}
                />
                <TextField
                  fullWidth
                  required
                  multiline
                  minRows={6}
                  label="Proposed factor payload (JSON)"
                  placeholder={`{\n  "factor_id": "EF:UK:road_freight_40t:2026:v1",\n  "co2e_per_unit": 0.105,\n  "unit": "tkm",\n  "source": "DEFRA",\n  "source_year": 2026\n}`}
                  value={toFactorPayloadJson}
                  onChange={(e) => setToFactorPayloadJson(e.target.value)}
                  sx={{ fontFamily: "monospace" }}
                />
              </Stack>
            )}

            {mode === "bulk" && (
              <TextField
                fullWidth
                required
                multiline
                minRows={3}
                label="Factor IDs to replace (one per line or comma-separated)"
                placeholder={"EF:UK:road_freight_40t:2025:v1\nEF:UK:road_freight_18t:2025:v1"}
                value={bulkIds}
                onChange={(e) => setBulkIds(e.target.value)}
              />
            )}

            <Stack direction="row" spacing={2}>
              <Button variant="contained" onClick={() => void run()} disabled={loading}>
                {loading ? "Simulating…" : "Simulate"}
              </Button>
              {report && (
                <Button
                  variant="contained"
                  color="success"
                  onClick={() => setPromoteOpen(true)}
                  disabled={loading || mode === "bulk"}
                  title={mode === "bulk" ? "Promote not available in bulk-replace mode" : undefined}
                >
                  Promote to Preview
                </Button>
              )}
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} role="alert" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {promotedReviewId && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Queued as review <code>{promotedReviewId}</code>. The methodology lead can approve from
          the Approval Queue page; the simulation is attached as evidence.
        </Alert>
      )}

      {report && (
        <>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid xs={12} md={3}>
              <StatCard
                label="Affected calculations"
                value={report.computation_count.toLocaleString()}
              />
            </Grid>
            <Grid xs={12} md={3}>
              <StatCard
                label="Affected customers"
                value={customerCount.toLocaleString()}
              />
            </Grid>
            <Grid xs={12} md={3}>
              <StatCard
                label="Affected inventories"
                value={inventoryCount.toLocaleString()}
              />
            </Grid>
            <Grid xs={12} md={3}>
              <StatCard
                label="Mean Δ%"
                value={meanDelta != null ? `${meanDelta.toFixed(2)}%` : "—"}
                color={meanDelta != null && Math.abs(meanDelta) > 5 ? "warning.main" : undefined}
              />
            </Grid>
            <Grid xs={12} md={3}>
              <StatCard
                label="Max Δ%"
                value={maxDelta != null ? `${maxDelta.toFixed(2)}%` : "—"}
                color={maxDelta != null && Math.abs(maxDelta) > 20 ? "error.main" : undefined}
              />
            </Grid>
            <Grid xs={12} md={3}>
              <StatCard
                label="Factors replaced"
                value={report.affected_factor_ids.length.toLocaleString()}
              />
            </Grid>
            <Grid xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline" color="text.secondary">
                    Δ% distribution
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <Histogram report={report} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Per-computation breakdown
                <Chip label={report.computations.length.toLocaleString()} size="small" sx={{ ml: 1 }} />
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 480 }}>
                <Table size="small" stickyHeader>
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
                        <TableCell align="right">
                          {c.old_value != null ? c.old_value.toFixed(4) : "—"}
                        </TableCell>
                        <TableCell align="right">
                          {c.new_value != null ? c.new_value.toFixed(4) : "—"}
                        </TableCell>
                        <TableCell align="right">
                          {c.delta_abs != null ? c.delta_abs.toFixed(4) : "—"}
                        </TableCell>
                        <TableCell align="right">
                          {c.delta_pct != null ? `${c.delta_pct.toFixed(2)}%` : "—"}
                        </TableCell>
                        <TableCell>
                          {c.evidence_bundle ? (
                            <Chip label={`${c.evidence_bundle.slice(0, 10)}…`} size="small" />
                          ) : (
                            "—"
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </>
      )}

      <Dialog open={promoteOpen} onClose={() => setPromoteOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Promote to Preview</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Submit a queue item proposing this factor be promoted to <code>preview</code>. The
            full simulation result is attached as evidence so the methodology lead can review the
            blast radius before approving.
          </Typography>
          <TextField
            fullWidth
            multiline
            minRows={3}
            label="Rationale"
            placeholder="e.g. DEFRA 2026 update; max delta 4.2% across 12 tenants."
            value={promoteRationale}
            onChange={(e) => setPromoteRationale(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPromoteOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => void promote()}
            disabled={promoting}
          >
            {promoting ? "Submitting…" : "Submit proposal"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FactorsImpactSimulator;
