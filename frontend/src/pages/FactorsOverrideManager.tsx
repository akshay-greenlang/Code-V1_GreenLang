/**
 * F7.6 — Customer override manager.
 *
 * Per-tenant factor overlay CRUD. Backing:
 *   GET /api/v1/factors/overlays?tenant_id=...
 *   POST /api/v1/factors/overlays
 *   DELETE /api/v1/factors/overlays/{id}
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
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

interface Overlay {
  id: string;
  tenant_id: string;
  factor_id: string;
  override_value: number;
  unit: string;
  valid_from: string;
  valid_to: string | null;
  active: boolean;
}

export function FactorsOverrideManager() {
  const [tenantId, setTenantId] = useState("");
  const [overlays, setOverlays] = useState<Overlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (!tenantId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ tenant_id: tenantId });
      const res = await fetch(`/api/v1/factors/overlays?${qs}`);
      if (!res.ok) throw new Error(`overlays ${res.status}`);
      const payload = await res.json();
      setOverlays((payload.overlays ?? []) as Overlay[]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const del = async (id: string) => {
    try {
      const res = await fetch(`/api/v1/factors/overlays/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(`delete ${res.status}`);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    if (tenantId) void load();
  }, []);

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Customer Override Manager</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Per-tenant factor overlays. Step 1 of the resolution cascade (Phase
        F3) — these always win over upstream factors.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <TextField
              fullWidth
              label="Tenant ID"
              value={tenantId}
              onChange={(e) => setTenantId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void load();
              }}
            />
            <Button variant="contained" onClick={() => void load()} disabled={!tenantId.trim()}>
              Load
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && <Alert severity="info" sx={{ mb: 2 }}>{error} — endpoint may not be wired yet.</Alert>}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Overlay ID</TableCell>
              <TableCell>Factor</TableCell>
              <TableCell align="right">Value</TableCell>
              <TableCell>Unit</TableCell>
              <TableCell>Valid from</TableCell>
              <TableCell>Valid to</TableCell>
              <TableCell>Active</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {overlays.map((o) => (
              <TableRow key={o.id}>
                <TableCell><code>{o.id.slice(0, 12)}…</code></TableCell>
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
                  <Button size="small" color="error" onClick={() => void del(o.id)}>
                    Delete
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {!loading && overlays.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No overlays for this tenant.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default FactorsOverrideManager;
