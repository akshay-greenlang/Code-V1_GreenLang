/**
 * F7.1 — Source ingestion console.
 *
 * Surfaces the source registry + watch pipeline + ingestion cadence so
 * operators can see the full upstream portfolio on one page. Backing
 * endpoints:
 *   - GET /api/v1/factors/status/summary      (Phase 5.3) — counts by source
 *   - GET /api/v1/factors/watch/status         (Phase 5.4) — per-source health
 *   - GET /api/v1/factors/search/facets        — source_id facet
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
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

interface SourceHealth {
  source_id: string;
  display_name?: string;
  cadence?: string;
  health: "healthy" | "stale" | "error" | "unknown";
  latest_timestamp: string | null;
  checks_in_window: number;
}

interface WatchStatus {
  source_count: number;
  health_counts: Record<string, number>;
  sources: SourceHealth[];
}

interface StatusSummary {
  edition_id: string;
  by_source: Array<{
    source_id: string;
    certified: number;
    preview: number;
    connector_only: number;
    deprecated: number;
    all: number;
  }>;
}

function HealthChip({ health }: { health: SourceHealth["health"] }) {
  const map: Record<string, "success" | "warning" | "error" | "default"> = {
    healthy: "success",
    stale: "warning",
    error: "error",
    unknown: "default",
  };
  return <Chip label={health} color={map[health]} size="small" variant="outlined" />;
}

export function FactorsSourceConsole() {
  const [watch, setWatch] = useState<WatchStatus | null>(null);
  const [summary, setSummary] = useState<StatusSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch("/api/v1/factors/watch/status").then((r) =>
        r.ok ? r.json() : Promise.reject(`watch ${r.status}`),
      ),
      fetch("/api/v1/factors/status/summary").then((r) =>
        r.ok ? r.json() : Promise.reject(`summary ${r.status}`),
      ),
    ])
      .then(([w, s]) => {
        if (cancelled) return;
        setWatch(w as WatchStatus);
        setSummary(s as StatusSummary);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <Box sx={{ p: 3 }}><LinearProgress /></Box>;
  if (error) return <Box sx={{ p: 3 }}><Alert severity="error">{error}</Alert></Box>;

  const summaryBySource = new Map(
    (summary?.by_source ?? []).map((s) => [s.source_id, s]),
  );

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Source Ingestion Console</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Live view of every upstream source: health of the last watch run,
        cadence, and how many factors each source contributes.
      </Typography>

      {watch && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Stack direction="row" spacing={1}>
              <Chip label={`Healthy: ${watch.health_counts.healthy ?? 0}`} color="success" />
              <Chip label={`Stale: ${watch.health_counts.stale ?? 0}`} color="warning" />
              <Chip label={`Error: ${watch.health_counts.error ?? 0}`} color="error" />
              <Chip label={`Unknown: ${watch.health_counts.unknown ?? 0}`} />
              <Chip label={`Total sources: ${watch.source_count}`} />
            </Stack>
          </CardContent>
        </Card>
      )}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Source</TableCell>
              <TableCell>Cadence</TableCell>
              <TableCell>Last check</TableCell>
              <TableCell>Health</TableCell>
              <TableCell align="right">Certified</TableCell>
              <TableCell align="right">Preview</TableCell>
              <TableCell align="right">Connector-only</TableCell>
              <TableCell align="right">Total</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(watch?.sources ?? []).map((s) => {
              const counts = summaryBySource.get(s.source_id);
              return (
                <TableRow key={s.source_id} hover>
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
                    {s.latest_timestamp
                      ? new Date(s.latest_timestamp).toLocaleString()
                      : "—"}
                  </TableCell>
                  <TableCell><HealthChip health={s.health} /></TableCell>
                  <TableCell align="right">{counts?.certified ?? "—"}</TableCell>
                  <TableCell align="right">{counts?.preview ?? "—"}</TableCell>
                  <TableCell align="right">{counts?.connector_only ?? "—"}</TableCell>
                  <TableCell align="right"><strong>{counts?.all ?? "—"}</strong></TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default FactorsSourceConsole;
