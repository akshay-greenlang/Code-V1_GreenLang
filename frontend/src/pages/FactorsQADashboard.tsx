/**
 * F7.3 — QA dashboard.
 *
 * Aggregates: review-queue depth, factors awaiting approval, factors
 * flagged by QA rules, stale verification, license-class exceptions.
 * Backing endpoint: GET /api/v1/factors/quality/dashboard (stub).
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

interface QASnapshot {
  review_queue_size: number;
  awaiting_approval: number;
  qa_flag_open: number;
  verification_stale: number;
  license_exceptions: number;
  last_run_at: string | null;
}

function StatCard({ label, value, severity }: { label: string; value: number; severity: "info" | "warning" | "error" | "success" }) {
  const colorMap: Record<string, string> = {
    info: "info.main",
    warning: "warning.main",
    error: "error.main",
    success: "success.main",
  };
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="overline" color="text.secondary">{label}</Typography>
        <Typography variant="h3" sx={{ color: colorMap[severity], fontWeight: 700 }}>
          {value.toLocaleString()}
        </Typography>
      </CardContent>
    </Card>
  );
}

export function FactorsQADashboard() {
  const [data, setData] = useState<QASnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/v1/factors/quality/dashboard")
      .then((r) => (r.ok ? r.json() : Promise.reject(`status ${r.status}`)))
      .then((d) => { if (!cancelled) setData(d as QASnapshot); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) return <Box sx={{ p: 3 }}><LinearProgress /></Box>;
  if (error || !data) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          QA dashboard not yet wired to a backend endpoint; shown as scaffolding.
          See `greenlang.factors.quality.review_workflow` for the existing
          review queue.
        </Alert>
        <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
          <StatCard label="Review queue" value={0} severity="info" />
          <StatCard label="Awaiting approval" value={0} severity="warning" />
          <StatCard label="QA flags open" value={0} severity="error" />
          <StatCard label="Verification stale" value={0} severity="warning" />
        </Stack>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Factors QA Dashboard</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Snapshot of open QA work. Last run: {data.last_run_at ?? "never"}.
      </Typography>
      <Grid container spacing={2}>
        <Grid xs={12} md={3}><StatCard label="Review queue" value={data.review_queue_size} severity="info" /></Grid>
        <Grid xs={12} md={3}><StatCard label="Awaiting approval" value={data.awaiting_approval} severity="warning" /></Grid>
        <Grid xs={12} md={3}><StatCard label="QA flags open" value={data.qa_flag_open} severity="error" /></Grid>
        <Grid xs={12} md={3}><StatCard label="Verification stale" value={data.verification_stale} severity="warning" /></Grid>
        <Grid xs={12} md={3}><StatCard label="License exceptions" value={data.license_exceptions} severity="error" /></Grid>
      </Grid>
    </Box>
  );
}

export default FactorsQADashboard;
