import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { listAgents, listPackTiers, listPolicyBundles } from "../api";
import type { AgentLifecycleRecord, PackTierRecord, PolicyBundleRecord } from "../types";

export function GovernancePage() {
  const [packs, setPacks] = useState<PackTierRecord[]>([]);
  const [agents, setAgents] = useState<AgentLifecycleRecord[]>([]);
  const [bundles, setBundles] = useState<PolicyBundleRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([listPackTiers(), listAgents(), listPolicyBundles()])
      .then(([packPayload, agentPayload, bundlePayload]) => {
        setPacks(packPayload);
        setAgents(agentPayload);
        setBundles(bundlePayload);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load governance data"));
  }, []);

  const packTierCounts = useMemo(() => {
    const counter = new Map<string, number>();
    for (const pack of packs) {
      counter.set(pack.tier, (counter.get(pack.tier) ?? 0) + 1);
    }
    return [...counter.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [packs]);

  const agentStateCounts = useMemo(() => {
    const counter = new Map<string, number>();
    for (const agent of agents) {
      counter.set(agent.state, (counter.get(agent.state) ?? 0) + 1);
    }
    return [...counter.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [agents]);

  return (
    <Stack spacing={2}>
      <Typography variant="h5">Governance Center</Typography>
      {error && <Alert severity="error">{error}</Alert>}
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Pack Tier Browser</Typography>
              {packTierCounts.map(([tier, count]) => (
                <Typography key={tier} variant="body2">{tier}: {count}</Typography>
              ))}
              {!packTierCounts.length && <Typography variant="body2">No pack records yet.</Typography>}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Agent Lifecycle Viewer</Typography>
              {agentStateCounts.map(([state, count]) => (
                <Typography key={state} variant="body2">{state}: {count}</Typography>
              ))}
              {!agentStateCounts.length && <Typography variant="body2">No agent lifecycle records yet.</Typography>}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Policy Inspector</Typography>
              {bundles.slice(0, 6).map((bundle) => (
                <Typography key={bundle.bundle} variant="body2">{bundle.bundle} ({bundle.bytes} bytes)</Typography>
              ))}
              {!bundles.length && <Typography variant="body2">No policy bundle metadata yet.</Typography>}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Stack>
  );
}
