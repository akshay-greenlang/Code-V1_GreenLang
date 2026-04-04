import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import {
  fetchConnectorHealth,
  fetchConnectorRegistry,
  fetchHealth,
  fetchReleaseTrainEvidence,
  listAgents,
  listPackTiers,
  listRuns
} from "../api";
import type { AgentLifecycleRecord, PackTierRecord, RunRecord } from "../types";

export function AdminPage() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null);
  const [releaseTrain, setReleaseTrain] = useState<Awaited<ReturnType<typeof fetchReleaseTrainEvidence>> | null>(
    null
  );
  const [connectors, setConnectors] = useState<Awaited<ReturnType<typeof fetchConnectorRegistry>> | null>(null);
  const [packs, setPacks] = useState<PackTierRecord[]>([]);
  const [agents, setAgents] = useState<AgentLifecycleRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [connectorProbeAt, setConnectorProbeAt] = useState<string | null>(null);
  const [probeBusy, setProbeBusy] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchHealth().catch(() => null),
      fetchReleaseTrainEvidence().catch(() => ({ available: false, evidence: null })),
      fetchConnectorRegistry().catch(() => null),
      listPackTiers().catch(() => []),
      listAgents().catch(() => []),
      listRuns().catch(() => [])
    ])
      .then(([h, rt, conn, pk, ag, rn]) => {
        setHealth(h);
        setReleaseTrain(rt);
        setConnectors(conn);
        setPacks(pk);
        setAgents(ag);
        setRuns(rn);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Admin load failed"))
      .finally(() => setLoading(false));
  }, []);

  const fallbackRuns = useMemo(
    () => runs.filter((r) => (r.execution_mode || "").toLowerCase() === "fallback").length,
    [runs]
  );

  return (
    <Stack spacing={2}>
      <Typography variant="h5">Admin Console</Typography>
      <Typography variant="body2" color="text.secondary">
        Release train evidence, connector registry health, and live governance signals for platform operators.
      </Typography>
      {loading && <LinearProgress aria-label="Loading admin data" />}
      {error && <Alert severity="error">{error}</Alert>}

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Shell health</Typography>
              {health ? (
                <Stack spacing={0.5} mt={1}>
                  <Typography variant="body2">Status: {health.status}</Typography>
                  <Typography variant="body2">Package version: {health.version}</Typography>
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Health endpoint unavailable.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Active sessions</Typography>
              <Typography variant="h4" sx={{ mt: 1 }}>
                {runs.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Fallback-mode runs (last fetch): {fallbackRuns}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Governance snapshot</Typography>
              <Typography variant="body2" mt={1}>
                Packs tracked: {packs.length}
              </Typography>
              <Typography variant="body2">Agents in registry: {agents.length}</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Release train (local evidence)</Typography>
              {!releaseTrain?.available && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  No docs/v2/RELEASE_TRAIN_LOCAL_EVIDENCE.json found on this server checkout.
                </Typography>
              )}
              {releaseTrain?.available && (
                <Stack spacing={1} mt={1}>
                  {(releaseTrain.cycle_summary ?? []).map((c) => (
                    <Stack key={String(c.cycle)} direction="row" spacing={1} alignItems="center">
                      <Chip
                        size="small"
                        label={c.all_passed ? "passed" : "issues"}
                        color={c.all_passed ? "success" : "warning"}
                      />
                      <Typography variant="body2">
                        {c.cycle} — {c.executed_at_utc ?? "unknown time"}
                      </Typography>
                    </Stack>
                  ))}
                  {!releaseTrain.cycle_summary?.length && (
                    <Typography variant="body2">Evidence file present but no cycles parsed.</Typography>
                  )}
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Connector reliability registry</Typography>
              {!connectors && (
                <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                  Could not load applications/connectors/v2_connector_registry.yaml.
                </Typography>
              )}
              {connectors && (
                <Stack spacing={0.75} mt={1}>
                  <Typography variant="caption" color="text.secondary">
                    Registry v{connectors.registry_version} — {connectors.connectors.length} connectors
                  </Typography>
                  {connectors.connectors.slice(0, 8).map((c) => (
                    <Typography key={c.connector_id} variant="body2">
                      {c.connector_id} · {c.app_id} · {(c.operational_status ?? "ok").toUpperCase()}
                      {c.slo_target_availability_pct != null ? ` · SLO ${c.slo_target_availability_pct}%` : ""} · read{" "}
                      {c.read_timeout_ms ?? "?"} ms
                    </Typography>
                  ))}
                  {connectors.connectors.length > 8 && (
                    <Typography variant="caption" color="text.secondary">
                      +{connectors.connectors.length - 8} more in registry file
                    </Typography>
                  )}
                  <Button
                    size="small"
                    variant="outlined"
                    disabled={probeBusy}
                    sx={{ mt: 1, alignSelf: "flex-start" }}
                    onClick={() => {
                      setProbeBusy(true);
                      fetchConnectorHealth()
                        .then((p) => setConnectorProbeAt(p.updated_at_utc))
                        .catch(() => setConnectorProbeAt(null))
                        .finally(() => setProbeBusy(false));
                    }}
                  >
                    Refresh live probe (stub)
                  </Button>
                  {connectorProbeAt && (
                    <Typography variant="caption" color="text.secondary">
                      Last probe refresh (UTC): {connectorProbeAt}
                    </Typography>
                  )}
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Stack>
  );
}
