/**
 * Track C-5 — OEM sub-tenant management.
 *
 * Wires to:
 *   GET    /v1/oem/subtenants
 *   POST   /v1/oem/subtenants
 *   DELETE /v1/oem/subtenants/{id}
 *   GET    /v1/oem/redistribution
 *
 * Lists every sub-tenant the OEM has provisioned, lets the operator
 * create a new sub-tenant (with the entitlement checkboxes constrained
 * to the parent OEM's redistribution grant), and revoke an existing
 * sub-tenant. Each row shows the sub-tenant's entitlements and active
 * status.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
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

interface SubTenant {
  id: string;
  oem_id: string;
  name: string;
  entitlements: string[];
  active: boolean;
  created_at: string;
  api_key?: string;
}

interface SubTenantsResponse {
  oem_id: string;
  count: number;
  subtenants: SubTenant[];
}

interface RedistributionGrantResponse {
  oem_id: string;
  parent_plan: string;
  allowed_classes: string[];
}

function readOemIdFromStorage(): string {
  try {
    return window.localStorage.getItem("greenlang.oem.id") ?? "";
  } catch {
    return "";
  }
}

export function OemSubTenants() {
  const [oemId, setOemId] = useState<string>(readOemIdFromStorage());
  const [subs, setSubs] = useState<SubTenant[]>([]);
  const [grant, setGrant] = useState<RedistributionGrantResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEntitlements, setNewEntitlements] = useState<Set<string>>(new Set());
  const [creating, setCreating] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);

  const headers = useMemo(
    () => (oemId ? { "X-OEM-Id": oemId } : undefined),
    [oemId]
  );

  const refresh = useCallback(async () => {
    if (!oemId) return;
    setLoading(true);
    setError(null);
    try {
      const [subsRes, grantRes] = await Promise.all([
        fetch(`/v1/oem/subtenants`, { headers }),
        fetch(`/v1/oem/redistribution`, { headers }),
      ]);
      if (!subsRes.ok) throw new Error(`subtenants: HTTP ${subsRes.status}`);
      if (!grantRes.ok) throw new Error(`redistribution: HTTP ${grantRes.status}`);
      const subsBody = (await subsRes.json()) as SubTenantsResponse;
      const grantBody = (await grantRes.json()) as RedistributionGrantResponse;
      setSubs(subsBody.subtenants);
      setGrant(grantBody);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [oemId, headers]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openCreate = () => {
    setNewName("");
    setNewEntitlements(new Set());
    setCreatedKey(null);
    setDialogOpen(true);
  };

  const toggleEntitlement = (cls: string) => {
    setNewEntitlements((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls);
      else next.add(cls);
      return next;
    });
  };

  const handleCreate = async () => {
    if (!oemId) return;
    setCreating(true);
    setError(null);
    try {
      const response = await fetch(`/v1/oem/subtenants`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-OEM-Id": oemId },
        body: JSON.stringify({
          name: newName.trim(),
          entitlements: Array.from(newEntitlements),
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      const body = (await response.json()) as SubTenant;
      setCreatedKey(body.api_key ?? null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (subId: string) => {
    if (!oemId) return;
    if (!window.confirm(`Revoke sub-tenant ${subId}?`)) return;
    try {
      const response = await fetch(`/v1/oem/subtenants/${subId}`, {
        method: "DELETE",
        headers: { "X-OEM-Id": oemId },
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 2 }}
      >
        <Typography variant="h4">OEM sub-tenants</Typography>
        <Stack direction="row" spacing={2} alignItems="center">
          <TextField
            label="OEM ID"
            size="small"
            value={oemId}
            onChange={(e) => {
              setOemId(e.target.value);
              try {
                window.localStorage.setItem(
                  "greenlang.oem.id",
                  e.target.value
                );
              } catch {
                /* ignore */
              }
            }}
            sx={{ minWidth: 280 }}
          />
          <Button variant="outlined" onClick={() => void refresh()}>
            Refresh
          </Button>
          <Button
            variant="contained"
            onClick={openCreate}
            disabled={!oemId || !grant}
          >
            New sub-tenant
          </Button>
        </Stack>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {grant && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Parent OEM grant ({grant.parent_plan})
            </Typography>
            <Box>
              {grant.allowed_classes.map((cls) => (
                <Chip
                  key={cls}
                  label={cls}
                  size="small"
                  sx={{ mr: 0.5, mb: 0.5 }}
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>ID</TableCell>
              <TableCell>Entitlements</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {subs.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={6}>
                  <Typography color="text.secondary" sx={{ py: 2 }}>
                    No sub-tenants provisioned yet.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {subs.map((s) => (
              <TableRow key={s.id}>
                <TableCell>{s.name}</TableCell>
                <TableCell>
                  <code style={{ fontSize: 12 }}>{s.id}</code>
                </TableCell>
                <TableCell>
                  {s.entitlements.length === 0 ? (
                    <Typography variant="caption" color="text.secondary">
                      (none)
                    </Typography>
                  ) : (
                    s.entitlements.map((cls) => (
                      <Chip
                        key={cls}
                        label={cls}
                        size="small"
                        sx={{ mr: 0.5 }}
                      />
                    ))
                  )}
                </TableCell>
                <TableCell>
                  <Chip
                    label={s.active ? "Active" : "Revoked"}
                    color={s.active ? "success" : "default"}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  {s.created_at ? new Date(s.created_at).toLocaleString() : "—"}
                </TableCell>
                <TableCell align="right">
                  {s.active && (
                    <Button
                      size="small"
                      color="error"
                      onClick={() => void handleRevoke(s.id)}
                    >
                      Revoke
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>New sub-tenant</DialogTitle>
        <DialogContent>
          {createdKey ? (
            <Stack spacing={2} sx={{ mt: 1 }}>
              <Alert severity="success">
                Sub-tenant created. Save the API key NOW — it will not be shown
                again.
              </Alert>
              <Typography variant="body2">
                <strong>API key:</strong>{" "}
                <code style={{ wordBreak: "break-all" }}>{createdKey}</code>
              </Typography>
            </Stack>
          ) : (
            <Stack spacing={2} sx={{ mt: 1 }}>
              <TextField
                label="Sub-tenant name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                fullWidth
                required
              />
              <Typography variant="subtitle2" color="text.secondary">
                Entitlements (subset of parent grant)
              </Typography>
              <FormGroup>
                {(grant?.allowed_classes ?? []).map((cls) => (
                  <FormControlLabel
                    key={cls}
                    control={
                      <Checkbox
                        checked={newEntitlements.has(cls)}
                        onChange={() => toggleEntitlement(cls)}
                      />
                    }
                    label={cls}
                  />
                ))}
              </FormGroup>
            </Stack>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            {createdKey ? "Close" : "Cancel"}
          </Button>
          {!createdKey && (
            <Button
              variant="contained"
              onClick={() => void handleCreate()}
              disabled={creating || !newName.trim()}
            >
              {creating ? "Creating..." : "Create"}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}
