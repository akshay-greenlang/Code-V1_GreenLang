import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Input from "@mui/material/Input";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { runApp } from "../api";
import type { AppKey, RunResponse } from "../types";

interface Props {
  app: AppKey;
  title: string;
  description: string;
}

export function WorkspacePage({ app, title, description }: Props) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [liveStatus, setLiveStatus] = useState("connecting");
  const [primaryFile, setPrimaryFile] = useState<File | undefined>(undefined);
  const [secondaryFile, setSecondaryFile] = useState<File | undefined>(undefined);

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
        const delayMs = Math.min(10_000, 1_000 * (2 ** (retries - 1)));
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
    return result.can_export ? "Export allowed" : "Export blocked by policy";
  }, [result]);

  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant="h5">{title}</Typography>
        <Typography color="text.secondary">{description}</Typography>
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
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="body2">Live status: {liveStatus}</Typography>
            <Button variant="contained" onClick={run} disabled={loading} aria-label={`Run ${app} workspace`}>
              Run Demo
            </Button>
          </Stack>
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
              <LinearProgress variant="determinate" value={progress} />
            </Box>
          )}
        </CardContent>
      </Card>
      {error && <Alert severity="error">{error}</Alert>}
      {result && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={8}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">Run {result.run_id}</Typography>
                <Typography>Status: {result.status}</Typography>
                <Typography>Policy: {policyVerdict}</Typography>
                <Typography mt={1}>Run graph: validate → compute → policy → export → audit</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">Artifacts</Typography>
                {(result.artifacts || []).map((artifact) => (
                  <Typography key={artifact} variant="body2">
                    {artifact}
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
