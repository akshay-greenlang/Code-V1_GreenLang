/**
 * F7.1 / Track B-5 — Source ingestion console.
 *
 * Wired to:
 *   GET  /v1/admin/sources             — registry + ingestion status
 *   POST /v1/admin/sources/ingest      — kick a re-ingest
 *   GET  /v1/admin/sources/{id}/runs   — per-source run history
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
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
  getSourceRuns,
  ingestSource,
  listSources,
  type IngestionRun,
  type SourceRecord,
} from "../lib/api/factorsClient";

function HealthChip({ health }: { health: SourceRecord["health"] }) {
  const map: Record<string, "success" | "warning" | "error" | "default"> = {
    healthy: "success",
    stale: "warning",
    error: "error",
    unknown: "default",
  };
  return (
    <Chip
      label={health ?? "unknown"}
      color={map[health ?? "unknown"]}
      size="small"
      variant="outlined"
    />
  );
}

function RunStatusChip({ status }: { status: IngestionRun["status"] }) {
  const map: Record<IngestionRun["status"], "success" | "warning" | "error" | "default"> = {
    succeeded: "success",
    running: "warning",
    queued: "default",
    failed: "error",
  };
  return <Chip label={status} color={map[status]} size="small" />;
}

interface SourceRowsProps {
  source: SourceRecord;
  expanded: boolean;
  onToggle: () => void;
  onIngest: () => void;
  busy: boolean;
  runs?: IngestionRun[];
  runsError?: string | null;
}

function SourceRows({ source: s, expanded, onToggle, onIngest, busy, runs, runsError }: SourceRowsProps) {
  return (
    <>
      <TableRow hover>
        <TableCell padding="checkbox">
          <IconButton
            size="small"
            onClick={onToggle}
            aria-label={expanded ? "Collapse runs" : "Expand runs"}
          >
            {expanded ? "▾" : "▸"}
          </IconButton>
        </TableCell>
        <TableCell>
          <Typography variant="body2"><code>{s.source_id}</code></Typography>
          {s.display_name && (
            <Typography variant="caption" color="text.secondary">
              {s.display_name}
            </Typography>
          )}
        </TableCell>
        <TableCell>{s.cadence ?? "—"}</TableCell>
        <TableCell>
          {s.latest_timestamp ? new Date(s.latest_timestamp).toLocaleString() : "—"}
        </TableCell>
        <TableCell><HealthChip health={s.health} /></TableCell>
        <TableCell align="right">
          {s.factor_count?.toLocaleString() ?? "—"}
        </TableCell>
        <TableCell>
          <Button size="small" variant="outlined" onClick={onIngest} disabled={busy}>
            {busy ? "Ingesting…" : "Re-ingest"}
          </Button>
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={7} sx={{ p: 0, borderBottom: expanded ? undefined : "none" }}>
          <Collapse in={expanded} unmountOnExit>
            <Box sx={{ p: 2, bgcolor: "action.hover" }}>
              <Typography variant="subtitle2" gutterBottom>
                Recent runs
              </Typography>
              {runsError && (
                <Alert severity="warning" sx={{ mb: 1 }}>
                  {runsError}
                </Alert>
              )}
              {!runs && !runsError && <LinearProgress />}
              {runs && (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Run ID</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Started</TableCell>
                      <TableCell>Finished</TableCell>
                      <TableCell align="right">Factors</TableCell>
                      <TableCell>Error</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {runs.map((r) => (
                      <TableRow key={r.run_id}>
                        <TableCell><code>{r.run_id.slice(0, 12)}…</code></TableCell>
                        <TableCell><RunStatusChip status={r.status} /></TableCell>
                        <TableCell>{new Date(r.started_at).toLocaleString()}</TableCell>
                        <TableCell>
                          {r.finished_at ? new Date(r.finished_at).toLocaleString() : "—"}
                        </TableCell>
                        <TableCell align="right">
                          {r.factor_count?.toLocaleString() ?? "—"}
                        </TableCell>
                        <TableCell>
                          {r.error ? (
                            <Typography variant="caption" color="error">
                              {r.error}
                            </Typography>
                          ) : (
                            "—"
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                    {runs.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={6} align="center">
                          <Typography variant="caption" color="text.secondary">
                            No runs recorded yet.
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              )}
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

export function FactorsSourceConsole() {
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyIngest, setBusyIngest] = useState<string | null>(null);
  const [openRuns, setOpenRuns] = useState<Record<string, boolean>>({});
  const [runsBySource, setRunsBySource] = useState<Record<string, IngestionRun[]>>({});
  const [runsError, setRunsError] = useState<Record<string, string | null>>({});

  const loadSources = async (initial = false) => {
    if (initial) setLoading(true);
    try {
      const res = await listSources();
      setSources(res.sources ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      if (initial) setLoading(false);
    }
  };

  useEffect(() => {
    void loadSources(true);
  }, []);

  const toggleRuns = async (sourceId: string) => {
    setOpenRuns((cur) => ({ ...cur, [sourceId]: !cur[sourceId] }));
    if (!runsBySource[sourceId] && !runsError[sourceId]) {
      try {
        const r = await getSourceRuns(sourceId);
        setRunsBySource((cur) => ({ ...cur, [sourceId]: r.runs ?? [] }));
        setRunsError((cur) => ({ ...cur, [sourceId]: null }));
      } catch (e) {
        const msg = e instanceof FactorsApiError ? e.userMessage : (e as Error).message;
        setRunsError((cur) => ({ ...cur, [sourceId]: msg }));
      }
    }
  };

  const triggerIngest = async (sourceId: string) => {
    setBusyIngest(sourceId);
    try {
      await ingestSource(sourceId);
      await loadSources();
      if (openRuns[sourceId]) {
        const r = await getSourceRuns(sourceId);
        setRunsBySource((cur) => ({ ...cur, [sourceId]: r.runs ?? [] }));
      }
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusyIngest(null);
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Source Ingestion Console
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Live view of every upstream source. Trigger a re-ingest, see the per-source run history,
        and check health badges.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Stack direction="row" spacing={1}>
            <Chip
              label={`Healthy: ${sources.filter((s) => s.health === "healthy").length}`}
              color="success"
            />
            <Chip
              label={`Stale: ${sources.filter((s) => s.health === "stale").length}`}
              color="warning"
            />
            <Chip
              label={`Error: ${sources.filter((s) => s.health === "error").length}`}
              color="error"
            />
            <Chip label={`Total: ${sources.length}`} />
          </Stack>
        </CardContent>
      </Card>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell />
              <TableCell>Source</TableCell>
              <TableCell>Cadence</TableCell>
              <TableCell>Last check</TableCell>
              <TableCell>Health</TableCell>
              <TableCell align="right">Factors</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sources.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No sources registered. Add one via the admin API.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {sources.map((s) => (
              <SourceRows
                key={s.source_id}
                source={s}
                expanded={!!openRuns[s.source_id]}
                onToggle={() => void toggleRuns(s.source_id)}
                onIngest={() => void triggerIngest(s.source_id)}
                busy={busyIngest === s.source_id}
                runs={runsBySource[s.source_id]}
                runsError={runsError[s.source_id]}
              />
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default FactorsSourceConsole;
