/**
 * O6 / W4-D — Entitlements admin (per-tenant plan + pack + data-class grants).
 *
 * Wired to:
 *   GET    /v1/admin/entitlements
 *   POST   /v1/admin/entitlements
 *   DELETE /v1/admin/entitlements/{tenant_id}
 *
 * Supports OEM sub-tenants: the grant has a nested `sub_tenants` list
 * that the UI renders inline. Cross-tenant blindness is enforced by the
 * backend (caller must have the `factors:admin` scope + parent-tenant
 * membership to edit a grant).
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
  listEntitlements,
  revokeEntitlement,
  upsertEntitlement,
  type EntitlementGrant,
} from "../lib/api/factorsClient";

interface GrantFormState {
  tenant_id: string;
  plan: string;
  packs: string;
  data_classes: string;
}

const EMPTY_FORM: GrantFormState = {
  tenant_id: "",
  plan: "",
  packs: "",
  data_classes: "",
};

function splitCsv(v: string): string[] {
  return v
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean);
}

export function FactorsEntitlementsAdmin() {
  const [grants, setGrants] = useState<EntitlementGrant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<GrantFormState>(EMPTY_FORM);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await listEntitlements();
      setGrants(r.grants ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const save = async () => {
    setBusy(true);
    try {
      await upsertEntitlement({
        tenant_id: form.tenant_id.trim(),
        plan: form.plan.trim() || undefined,
        packs: splitCsv(form.packs),
        data_classes: splitCsv(form.data_classes),
      });
      setOpen(false);
      setForm(EMPTY_FORM);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (tenantId: string) => {
    if (!window.confirm(`Revoke all entitlements for ${tenantId}?`)) return;
    try {
      await revokeEntitlement(tenantId);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    }
  };

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Entitlements Admin
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Grant / revoke plan + premium-pack + data-class access per tenant.
        Sub-tenants inherit unless explicitly overridden.
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <Button
          variant="contained"
          onClick={() => {
            setForm(EMPTY_FORM);
            setOpen(true);
          }}
          data-testid="entitlements-add"
        >
          + New grant
        </Button>
        <Button variant="outlined" onClick={() => void load()} disabled={loading}>
          Refresh
        </Button>
      </Stack>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="Entitlement grants by tenant">
              <TableHead>
                <TableRow>
                  <TableCell>Tenant</TableCell>
                  <TableCell>Plan</TableCell>
                  <TableCell>Packs</TableCell>
                  <TableCell>Data classes</TableCell>
                  <TableCell>Sub-tenants</TableCell>
                  <TableCell>Updated</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {grants.map((g) => (
                  <TableRow key={g.tenant_id}>
                    <TableCell><code>{g.tenant_id}</code></TableCell>
                    <TableCell>
                      {g.plan ? <Chip size="small" label={g.plan} color="primary" /> : "—"}
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {(g.packs ?? []).map((p) => (
                          <Chip key={p} size="small" variant="outlined" label={p} />
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {(g.data_classes ?? []).map((d) => (
                          <Chip key={d} size="small" color="info" variant="outlined" label={d} />
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      {(g.sub_tenants ?? []).length > 0 ? (
                        <Stack spacing={0.5}>
                          {(g.sub_tenants ?? []).map((s) => (
                            <Typography key={s.tenant_id} variant="caption">
                              <code>{s.tenant_id}</code>{" "}
                              {s.plan && <Chip size="small" label={s.plan} />}
                            </Typography>
                          ))}
                        </Stack>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell>
                      {g.updated_at ? new Date(g.updated_at).toLocaleString() : "—"}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        color="error"
                        onClick={() => void revoke(g.tenant_id)}
                        data-testid={`entitlements-revoke-${g.tenant_id}`}
                      >
                        Revoke
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
                {!loading && grants.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No entitlements granted.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>New / update grant</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              required
              label="Tenant ID"
              value={form.tenant_id}
              onChange={(e) => setForm((f) => ({ ...f, tenant_id: e.target.value }))}
            />
            <TextField
              label="Plan"
              placeholder="community / pro / enterprise"
              value={form.plan}
              onChange={(e) => setForm((f) => ({ ...f, plan: e.target.value }))}
            />
            <TextField
              label="Packs (comma / space separated)"
              placeholder="electricity-premium freight-premium"
              value={form.packs}
              onChange={(e) => setForm((f) => ({ ...f, packs: e.target.value }))}
              data-testid="entitlements-packs-input"
            />
            <TextField
              label="Data classes"
              placeholder="pii-redacted iot-raw"
              value={form.data_classes}
              onChange={(e) => setForm((f) => ({ ...f, data_classes: e.target.value }))}
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => void save()}
            disabled={busy || !form.tenant_id.trim()}
            data-testid="entitlements-save"
          >
            {busy ? "Saving…" : "Save grant"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FactorsEntitlementsAdmin;
