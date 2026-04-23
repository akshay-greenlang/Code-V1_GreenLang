/**
 * O7 / W4-D — Audit bundle export.
 *
 * Submit a job that packages: factors, source citations, method-pack
 * documents, and signed receipts into a downloadable ZIP.
 *
 * Wired to:
 *   POST /v1/admin/audit-bundles/export
 *   GET  /v1/admin/audit-bundles/jobs/{job_id}
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import FormControlLabel from "@mui/material/FormControlLabel";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  getAuditBundleJob,
  submitAuditBundleExport,
  type AuditBundleJob,
} from "../lib/api/factorsClient";

function statusColor(s: string): "success" | "warning" | "default" | "error" {
  if (s === "completed") return "success";
  if (s === "running") return "warning";
  if (s === "failed") return "error";
  return "default";
}

export function FactorsAuditBundleExport() {
  const [factorIds, setFactorIds] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [includeMethodPacks, setIncludeMethodPacks] = useState(true);
  const [includeReceipts, setIncludeReceipts] = useState(true);
  const [edition, setEdition] = useState("");
  const [job, setJob] = useState<AuditBundleJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll the job every 2s while it's not terminal.
  useEffect(() => {
    if (!job) return;
    if (job.status === "completed" || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getAuditBundleJob(job.job_id);
        setJob(next);
      } catch (e) {
        setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job]);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const ids = factorIds
        .split(/[\n,\s]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      const r = await submitAuditBundleExport({
        factor_ids: ids.length > 0 ? ids : undefined,
        tenant_id: tenantId.trim() || undefined,
        include_method_packs: includeMethodPacks,
        include_signed_receipts: includeReceipts,
        edition: edition.trim() || undefined,
      });
      setJob(r);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 900, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Audit Bundle Export
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Build a downloadable ZIP containing factors + source citations +
        method-pack documents + signed receipts. Paste factor IDs (one
        per line) or leave blank to export every factor visible to the
        caller's tenant.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            <TextField
              label="Factor IDs (optional)"
              multiline
              minRows={3}
              placeholder="EF:IN:grid_electricity:2026:v1"
              value={factorIds}
              onChange={(e) => setFactorIds(e.target.value)}
              data-testid="audit-bundle-factor-ids"
            />
            <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
              <TextField
                label="Tenant ID (optional)"
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                sx={{ flex: 1 }}
              />
              <TextField
                label="Edition (optional)"
                placeholder="v1.0.0"
                value={edition}
                onChange={(e) => setEdition(e.target.value)}
                sx={{ flex: 1 }}
              />
            </Stack>
            <Stack direction="row" spacing={2}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={includeMethodPacks}
                    onChange={(e) => setIncludeMethodPacks(e.target.checked)}
                  />
                }
                label="Include method-pack docs"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={includeReceipts}
                    onChange={(e) => setIncludeReceipts(e.target.checked)}
                  />
                }
                label="Include signed receipts"
              />
            </Stack>
            <Button
              variant="contained"
              onClick={() => void submit()}
              disabled={submitting}
              data-testid="audit-bundle-submit"
            >
              {submitting ? "Submitting…" : "Export bundle"}
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
        <Card data-testid="audit-bundle-job-panel">
          <CardContent>
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="h6">
                  Job <code>{job.job_id}</code>
                </Typography>
                <Chip
                  size="small"
                  label={job.status}
                  color={statusColor(job.status)}
                  data-testid="audit-bundle-job-status"
                />
              </Stack>
              {job.status === "running" && <LinearProgress />}
              {job.included_factors !== undefined && (
                <Typography variant="body2" color="text.secondary">
                  Bundling {job.included_factors.toLocaleString()} factors.
                </Typography>
              )}
              {job.error_message && (
                <Alert severity="error">{job.error_message}</Alert>
              )}
              {job.status === "completed" && job.download_url && (
                <Button
                  variant="contained"
                  color="success"
                  href={job.download_url}
                  target="_blank"
                  rel="noreferrer"
                  data-testid="audit-bundle-download"
                >
                  Download ZIP
                </Button>
              )}
              {job.created_at && (
                <Typography variant="caption" color="text.secondary">
                  Submitted {new Date(job.created_at).toLocaleString()}
                  {job.completed_at &&
                    ` · completed ${new Date(job.completed_at).toLocaleString()}`}
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default FactorsAuditBundleExport;
