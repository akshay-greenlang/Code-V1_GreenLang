import { useEffect, useMemo, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import Input from "@mui/material/Input";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { RunStatusChip, ShellErrorAlert, apiStatusChipFromResponse, errorEnvelopeFromApi } from "@greenlang/shell-ui";
import { runApp, runArtifactUrl } from "../api";
import { RunGraphDag } from "../components/RunGraphDag";
import type { StageId } from "../pipelineStages";
import type { AppKey, RunErrorEnvelope, RunResponse } from "../types";
import { workspaceByApp } from "../workspaceConfig";

interface Props {
  app: AppKey;
  title: string;
  description: string;
}

export function WorkspacePage({ app, title, description }: Props) {
  const cfg = workspaceByApp[app];
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [liveStatus, setLiveStatus] = useState("connecting");
  const [primaryFile, setPrimaryFile] = useState<File | undefined>(undefined);
  const [secondaryFile, setSecondaryFile] = useState<File | undefined>(undefined);
  const [selectedStage, setSelectedStage] = useState<StageId | null>("validate");
  const [checklist, setChecklist] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let retries = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setLiveStatus(retries === 0 ? "connecting" : "reconnecting");
      source = new EventSource("/api/v1/stream/runs");
      source.onopen = () => {
        retries = 0;
        setLiveStatus("live");
      };
      source.onmessage = (evt) => {
        try {
          const payload = JSON.parse(evt.data) as { status?: string };
          setLiveStatus(payload.status || "live");
        } catch {
          setLiveStatus("live");
        }
      };
      source.onerror = () => {
        source?.close();
        if (disposed) return;
        retries += 1;
        if (retries >= 5) {
          setLiveStatus("degraded");
          return;
        }
        setLiveStatus("reconnecting");
        const delayMs = Math.min(10_000, 1_000 * 2 ** (retries - 1));
        reconnectTimer = setTimeout(connect, delayMs);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      source?.close();
    };
  }, []);

  const run = async () => {
    let progressTimer: ReturnType<typeof setInterval> | undefined;
    setLoading(true);
    setError(null);
    setProgress(15);
    try {
      progressTimer = setInterval(() => setProgress((v) => Math.min(95, v + 8)), 250);
      const data = await runApp(app, primaryFile, secondaryFile);
      setProgress(100);
      setResult(data);
      setLiveStatus("completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setLiveStatus("failed");
    } finally {
      if (progressTimer) clearInterval(progressTimer);
      setLoading(false);
    }
  };

  const policyVerdict = useMemo(() => {
    if (!result) return "No run yet";
    if (result.can_export === false) return "Export blocked by policy";
    if (result.warnings?.length) return "Export allowed with warnings — review before submission";
    return "Export allowed";
  }, [result]);

  const runLifecycle = result?.run_state;

  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant="h5">{title}</Typography>
        <Typography color="text.secondary">{description}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {cfg.regulatoryNotes}
        </Typography>
        <Link component={RouterLink} to={`/runs?app=${app}`} sx={{ mt: 1, display: "inline-block" }} variant="body2">
          Open Run Center for this app
        </Link>
      </Box>
      <Card variant="outlined">
        <CardContent>
          {(liveStatus === "reconnecting" || liveStatus === "degraded") && (
            <Alert severity={liveStatus === "degraded" ? "warning" : "info"} sx={{ mb: 2 }}>
              {liveStatus === "degraded"
                ? "Live updates degraded. Showing last known status."
                : "Reconnecting live status channel..."}
            </Alert>
          )}
          <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
            <Typography variant="body2" aria-live="polite" aria-atomic="true">
              Live status: {liveStatus}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button variant="outlined" onClick={run} disabled={loading} aria-label={`Retry ${app} workspace run`}>
                Retry run
              </Button>
              <Button variant="contained" onClick={run} disabled={loading} aria-label={`Run ${app} workspace`}>
                Run demo
              </Button>
            </Stack>
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            Primary file: {cfg.primaryFileHint}
            {cfg.secondaryFileHint ? ` Secondary: ${cfg.secondaryFileHint}` : ""}
          </Typography>
          <Stack direction="row" spacing={1} mt={2}>
            <Input
              type="file"
              inputProps={{ "aria-label": `${app} input file` }}
              onChange={(event) => setPrimaryFile((event.target as HTMLInputElement).files?.[0])}
            />
            {app === "cbam" && (
              <Input
                type="file"
                inputProps={{ "aria-label": "cbam imports file" }}
                onChange={(event) => setSecondaryFile((event.target as HTMLInputElement).files?.[0])}
              />
            )}
          </Stack>
          {(loading || progress > 0) && (
            <Box mt={2}>
              <LinearProgress variant="determinate" value={progress} aria-label="Run progress" />
            </Box>
          )}
        </CardContent>
      </Card>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Regulatory checklist</Typography>
          <Stack sx={{ mt: 1 }}>
            {cfg.checklist.map((item) => (
              <FormControlLabel
                key={item.id}
                control={
                  <Checkbox
                    checked={!!checklist[item.id]}
                    onChange={(_, v) => setChecklist((prev) => ({ ...prev, [item.id]: v }))}
                    inputProps={{ "aria-label": item.label }}
                  />
                }
                label={item.label}
              />
            ))}
          </Stack>
        </CardContent>
      </Card>
      {error && (
        <Alert severity="error" action={<Button onClick={run}>Retry</Button>}>
          {error}
        </Alert>
      )}
      {result?.run_state === "blocked" && (
        <Alert severity="warning">Run finished but export is blocked by policy or gate. Review artifacts and policy output.</Alert>
      )}
      {result?.run_state === "partial_success" && (result.warnings?.length ?? 0) > 0 && (
        <Alert severity="info">
          Partial success with warnings:
          <Box component="ul" sx={{ pl: 2, mb: 0 }}>
            {(result.warnings ?? []).map((w) => (
              <li key={w}>{w}</li>
            ))}
          </Box>
        </Alert>
      )}
      {result &&
        (() => {
          const env = errorEnvelopeFromApi(result.error_envelope as RunErrorEnvelope | null | undefined, result.errors);
          return env ? <ShellErrorAlert key="run-errors" envelope={env} /> : null;
        })()}
      {result && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={8}>
            <Card variant="outlined">
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap" useFlexGap>
                  <Typography variant="h6">Run {result.run_id}</Typography>
                  <RunStatusChip chip={apiStatusChipFromResponse(result.status_chip)} runState={runLifecycle} />
                </Stack>
                <Typography>Status: {result.status}</Typography>
                <Typography>Lifecycle: {runLifecycle ?? "unknown"}</Typography>
                <Typography>Policy: {policyVerdict}</Typography>
                <Box sx={{ mt: 2 }}>
                  <RunGraphDag
                    runId={result.run_id}
                    artifacts={result.artifacts || []}
                    runState={runLifecycle}
                    selectedStage={selectedStage}
                    onSelectStage={setSelectedStage}
                    artifactUrl={(path) => runArtifactUrl(result.run_id, path)}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">Artifacts</Typography>
                {(result.artifacts || []).map((artifact) => (
                  <Typography key={artifact} variant="body2">
                    <Link href={runArtifactUrl(result.run_id, artifact)} color="secondary">
                      {artifact}
                    </Link>
                  </Typography>
                ))}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Stack>
  );
}
