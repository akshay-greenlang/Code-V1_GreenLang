/**
 * O11 / W4-D — Batch-resolve jobs (CSV/JSON upload).
 *
 * Wired to:
 *   POST /v1/factors/resolve/batch
 *   GET  /v1/factors/jobs/{job_id}
 *
 * Accepts pasted JSON array or a CSV file (converted client-side into
 * the same request shape). Progress bar polls the status endpoint every
 * 2 s until terminal. On completion exposes the result URL + errors URL.
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  getBatchJob,
  submitBatchResolveJob,
  type BatchJobRecord,
} from "../lib/api/factorsClient";

/** Parse a minimal CSV where the first row is the header. */
function csvToRequests(csv: string): Array<Record<string, unknown>> {
  const lines = csv.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((row) => {
    const cells = row.split(",").map((c) => c.trim());
    const obj: Record<string, unknown> = {};
    headers.forEach((h, i) => {
      obj[h] = cells[i] ?? "";
    });
    return obj;
  });
}

function statusColor(s?: string): "success" | "warning" | "error" | "default" {
  if (s === "completed") return "success";
  if (s === "running") return "warning";
  if (s === "failed" || s === "cancelled") return "error";
  return "default";
}

export function FactorsBatchJobs() {
  const [mode, setMode] = useState<"json" | "csv">("json");
  const [payload, setPayload] = useState("");
  const [job, setJob] = useState<BatchJobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!job) return;
    if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
      return;
    }
    const t = window.setInterval(async () => {
      try {
        const next = await getBatchJob(job.job_id);
        setJob(next);
      } catch (e) {
        setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      }
    }, 2000);
    return () => window.clearInterval(t);
  }, [job]);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      let requests: Array<Record<string, unknown>>;
      if (mode === "json") {
        const parsed = JSON.parse(payload);
        if (!Array.isArray(parsed)) throw new Error("JSON payload must be an array.");
        requests = parsed as Array<Record<string, unknown>>;
      } else {
        requests = csvToRequests(payload);
        if (requests.length === 0) throw new Error("CSV produced 0 rows.");
      }
      const r = await submitBatchResolveJob(requests);
      setJob(r);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    const txt = await file.text();
    setPayload(txt);
    setMode(file.name.toLowerCase().endsWith(".json") ? "json" : "csv");
  };

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1000, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Batch Resolve Jobs
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Submit a CSV or JSON array of resolution requests. The server
        returns a job_id; progress polls every 2 s until completion. On
        success you can download results and the per-row errors file.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            <Stack direction="row" spacing={2} alignItems="center">
              <Button
                variant={mode === "json" ? "contained" : "outlined"}
                onClick={() => setMode("json")}
              >
                JSON
              </Button>
              <Button
                variant={mode === "csv" ? "contained" : "outlined"}
                onClick={() => setMode("csv")}
              >
                CSV
              </Button>
              <Button variant="outlined" component="label">
                Upload file
                <input
                  type="file"
                  accept=".csv,.json,application/json,text/csv"
                  hidden
                  onChange={(e) => void onFile(e.target.files?.[0])}
                  data-testid="batch-jobs-file"
                />
              </Button>
            </Stack>
            <TextField
              label={mode === "json" ? "JSON array" : "CSV payload"}
              placeholder={
                mode === "json"
                  ? '[ { "activity": "grid_electricity", "method_profile": "default", "jurisdiction": "IN" } ]'
                  : "activity,method_profile,jurisdiction\ngrid_electricity,default,IN"
              }
              multiline
              minRows={8}
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              data-testid="batch-jobs-payload"
              sx={{ fontFamily: "monospace" }}
            />
            <Button
              variant="contained"
              onClick={() => void submit()}
              disabled={submitting || !payload.trim()}
              data-testid="batch-jobs-submit"
            >
              {submitting ? "Submitting…" : "Submit batch"}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {job && (
        <Card data-testid="batch-jobs-status-panel">
          <CardContent>
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="h6">
                  Job <code>{job.job_id}</code>
                </Typography>
                <Chip
                  size="small"
                  color={statusColor(job.status)}
                  label={job.status}
                  data-testid="batch-jobs-status"
                />
                {job.progress_percent !== undefined && (
                  <Chip size="small" label={`${job.progress_percent.toFixed(0)}%`} />
                )}
              </Stack>
              {(job.status === "running" || job.status === "queued") && (
                <LinearProgress
                  variant={job.progress_percent !== undefined ? "determinate" : "indeterminate"}
                  value={job.progress_percent ?? 0}
                />
              )}
              {job.total_items !== undefined && (
                <Typography variant="body2" color="text.secondary">
                  {job.processed_items?.toLocaleString() ?? 0} / {job.total_items.toLocaleString()} items processed
                </Typography>
              )}
              {job.error_message && <Alert severity="error">{job.error_message}</Alert>}
              <Stack direction="row" spacing={2}>
                {job.results_url && (
                  <Button
                    variant="contained"
                    color="success"
                    href={job.results_url}
                    target="_blank"
                    rel="noreferrer"
                    data-testid="batch-jobs-download-results"
                  >
                    Download results
                  </Button>
                )}
                {job.errors_url && (
                  <Button
                    variant="outlined"
                    color="error"
                    href={job.errors_url}
                    target="_blank"
                    rel="noreferrer"
                    data-testid="batch-jobs-download-errors"
                  >
                    Download errors
                  </Button>
                )}
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default FactorsBatchJobs;
