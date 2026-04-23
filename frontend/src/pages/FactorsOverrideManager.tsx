/**
 * F7.6 / Track B-5 — Customer override manager.
 *
 * Wired to:
 *   GET    /v1/admin/overrides              — list (optionally filtered by tenant)
 *   POST   /v1/admin/overrides              — create
 *   DELETE /v1/admin/overrides/{id}         — soft-delete
 *
 * Per-tenant factor overlays. Always wins over upstream factors in the
 * resolution cascade (Phase F3, step 1).
 */
import { useEffect, useState } from "react";
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
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  createOverride,
  deleteOverride,
  listOverrides,
  type OverrideRecord,
} from "../lib/api/factorsClient";

interface CreateForm {
  tenant_id: string;
  factor_id: string;
  override_value: string;
  unit: string;
  valid_from: string;
  valid_to: string;
  rationale: string;
}

const EMPTY_FORM: CreateForm = {
  tenant_id: "",
  factor_id: "",
  override_value: "",
  unit: "",
  valid_from: new Date().toISOString().slice(0, 10),
  valid_to: "",
  rationale: "",
};

/**
 * W4-D — Cross-tenant blindness: the caller's tenant id is read from the
 * JWT (best-effort local parse) OR from localStorage `gl.auth.tenant`.
 * When present, the tenant filter is locked to that value so the UI
 * cannot leak other tenants' overrides even if the backend mistakenly
 * permits it.
 */
function readCallerTenant(): string | null {
  try {
    const explicit = window.localStorage.getItem("gl.auth.tenant");
    if (explicit && explicit.trim().length > 0) return explicit.trim();
    const token = window.localStorage.getItem("gl.auth.token");
    if (!token) return null;
    const parts = token.split(".");
    if (parts.length < 2) return null;
    const payload = parts[1];
    const pad = (4 - (payload.length % 4)) % 4;
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat(pad);
    const json = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("utf-8");
    const claims = JSON.parse(json) as { tenant_id?: string; tenant?: string };
    return claims.tenant_id ?? claims.tenant ?? null;
  } catch {
    return null;
  }
}

export function FactorsOverrideManager() {
  const callerTenant = readCallerTenant();
  const [tenantFilter, setTenantFilter] = useState(callerTenant ?? "");
  const [overrides, setOverrides] = useState<OverrideRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [form, setForm] = useState<CreateForm>(EMPTY_FORM);
  const [creating, setCreating] = useState(false);
  const [busyDelete, setBusyDelete] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await listOverrides({ tenant_id: tenantFilter.trim() || undefined });
      setOverrides(r.overrides ?? []);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    // Re-load whenever the filter changes via the Load button only — avoid
    // hammering the API on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onCreate = async () => {
    setCreating(true);
    setError(null);
    try {
      const valNum = Number(form.override_value);
      if (!Number.isFinite(valNum)) {
        throw new Error("Override value must be a number.");
      }
      await createOverride({
        tenant_id: form.tenant_id.trim(),
        factor_id: form.factor_id.trim(),
        override_value: valNum,
        unit: form.unit.trim(),
        valid_from: form.valid_from,
        valid_to: form.valid_to.trim() || null,
        rationale: form.rationale.trim() || undefined,
      });
      setCreateOpen(false);
      setForm(EMPTY_FORM);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const onDelete = async (id: string) => {
    setBusyDelete(id);
    try {
      await deleteOverride(id);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusyDelete(null);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Customer Override Manager
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Per-tenant factor overlays. Step 1 of the resolution cascade — these always win over
        upstream factors.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="center">
            <TextField
              fullWidth
              label={callerTenant ? "Tenant (locked — cross-tenant blindness)" : "Filter by tenant ID"}
              value={tenantFilter}
              onChange={(e) => {
                // Cross-tenant blindness: operators cannot type another
                // tenant id when the UI resolved a caller tenant from
                // the session.
                if (callerTenant) return;
                setTenantFilter(e.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") void load();
              }}
              InputProps={{ readOnly: Boolean(callerTenant) }}
              data-testid="override-tenant-filter"
              helperText={
                callerTenant
                  ? `Tenant ${callerTenant} — locked by session for cross-tenant blindness.`
                  : undefined
              }
            />
            <Button variant="outlined" onClick={() => void load()} disabled={loading}>
              {loading ? "Loading…" : "Load"}
            </Button>
            <Button variant="contained" onClick={() => setCreateOpen(true)}>
              + New override
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Override ID</TableCell>
              <TableCell>Tenant</TableCell>
              <TableCell>Factor</TableCell>
              <TableCell align="right">Value</TableCell>
              <TableCell>Unit</TableCell>
              <TableCell>Valid from</TableCell>
              <TableCell>Valid to</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {overrides.map((o) => (
              <TableRow key={o.id}>
                <TableCell><code>{o.id.slice(0, 12)}…</code></TableCell>
                <TableCell><code>{o.tenant_id}</code></TableCell>
                <TableCell><code>{o.factor_id}</code></TableCell>
                <TableCell align="right">{o.override_value}</TableCell>
                <TableCell>{o.unit}</TableCell>
                <TableCell>{o.valid_from}</TableCell>
                <TableCell>{o.valid_to ?? "—"}</TableCell>
                <TableCell>
                  <Chip
                    label={o.active ? "active" : "inactive"}
                    color={o.active ? "success" : "default"}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Button
                    size="small"
                    color="error"
                    onClick={() => void onDelete(o.id)}
                    disabled={busyDelete === o.id}
                  >
                    {busyDelete === o.id ? "…" : "Delete"}
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {!loading && overrides.length === 0 && (
              <TableRow>
                <TableCell colSpan={9} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No overrides configured{tenantFilter ? ` for tenant ${tenantFilter}` : ""}.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>New override</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              required
              label="Tenant ID"
              value={form.tenant_id}
              onChange={(e) => setForm((f) => ({ ...f, tenant_id: e.target.value }))}
            />
            <TextField
              required
              label="Factor ID"
              value={form.factor_id}
              onChange={(e) => setForm((f) => ({ ...f, factor_id: e.target.value }))}
            />
            <Stack direction="row" spacing={2}>
              <TextField
                required
                label="Override value"
                value={form.override_value}
                onChange={(e) => setForm((f) => ({ ...f, override_value: e.target.value }))}
                sx={{ flex: 1 }}
              />
              <TextField
                required
                label="Unit"
                value={form.unit}
                onChange={(e) => setForm((f) => ({ ...f, unit: e.target.value }))}
                sx={{ flex: 1 }}
              />
            </Stack>
            <Stack direction="row" spacing={2}>
              <TextField
                required
                label="Valid from"
                type="date"
                InputLabelProps={{ shrink: true }}
                value={form.valid_from}
                onChange={(e) => setForm((f) => ({ ...f, valid_from: e.target.value }))}
                sx={{ flex: 1 }}
              />
              <TextField
                label="Valid to (optional)"
                type="date"
                InputLabelProps={{ shrink: true }}
                value={form.valid_to}
                onChange={(e) => setForm((f) => ({ ...f, valid_to: e.target.value }))}
                sx={{ flex: 1 }}
              />
            </Stack>
            <TextField
              label="Rationale"
              multiline
              minRows={2}
              value={form.rationale}
              onChange={(e) => setForm((f) => ({ ...f, rationale: e.target.value }))}
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => void onCreate()}
            disabled={
              creating ||
              !form.tenant_id.trim() ||
              !form.factor_id.trim() ||
              !form.override_value.trim() ||
              !form.unit.trim() ||
              !form.valid_from
            }
          >
            {creating ? "Creating…" : "Create"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FactorsOverrideManager;
