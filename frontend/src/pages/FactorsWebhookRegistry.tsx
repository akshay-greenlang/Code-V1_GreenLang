/**
 * O10 / W4-D — Webhook registry (per-tenant).
 *
 * Wired to:
 *   GET    /v1/admin/webhooks
 *   POST   /v1/admin/webhooks
 *   DELETE /v1/admin/webhooks/{id}
 *   GET    /v1/admin/webhooks/{id}/deliveries
 *   POST   /v1/admin/webhooks/{id}/deliveries/{deliveryId}/replay
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
  createWebhook,
  deleteWebhook,
  listWebhookDeliveries,
  listWebhooks,
  replayWebhookDelivery,
  type WebhookDelivery,
  type WebhookRegistration,
} from "../lib/api/factorsClient";

function deliveryColor(s: string): "success" | "warning" | "error" | "default" {
  if (s === "success") return "success";
  if (s === "retrying") return "warning";
  if (s === "failed") return "error";
  return "default";
}

export function FactorsWebhookRegistry() {
  const [webhooks, setWebhooks] = useState<WebhookRegistration[]>([]);
  const [deliveries, setDeliveries] = useState<Record<string, WebhookDelivery[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState("");
  const [events, setEvents] = useState("factor.updated,factor.resolved");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const r = await listWebhooks();
      setWebhooks(r.webhooks ?? []);
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

  const openDeliveries = async (webhookId: string) => {
    try {
      const r = await listWebhookDeliveries(webhookId);
      setDeliveries((cur) => ({ ...cur, [webhookId]: r.deliveries ?? [] }));
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    }
  };

  const replay = async (webhookId: string, deliveryId: string) => {
    try {
      await replayWebhookDelivery(webhookId, deliveryId);
      await openDeliveries(webhookId);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    }
  };

  const create = async () => {
    setBusy(true);
    try {
      await createWebhook({
        url: url.trim(),
        events: events.split(",").map((e) => e.trim()).filter(Boolean),
      });
      setOpen(false);
      setUrl("");
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("Delete webhook?")) return;
    try {
      await deleteWebhook(id);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    }
  };

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Webhook Registry
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Register per-tenant webhook URLs. Inspect delivery history, retry
        failed deliveries, and see the HMAC secret hint.
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <Button variant="contained" onClick={() => setOpen(true)} data-testid="webhooks-add">
          + Register webhook
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
            <Table size="small" aria-label="Webhook registrations">
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>URL</TableCell>
                  <TableCell>Events</TableCell>
                  <TableCell>Enabled</TableCell>
                  <TableCell>Secret hint</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {webhooks.map((w) => (
                  <TableRow key={w.id}>
                    <TableCell><code>{w.id.slice(0, 8)}…</code></TableCell>
                    <TableCell sx={{ maxWidth: 280, wordBreak: "break-all" }}>
                      <code>{w.url}</code>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {w.events.map((e) => (
                          <Chip key={e} size="small" variant="outlined" label={e} />
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={w.enabled ? "success" : "default"}
                        label={w.enabled ? "enabled" : "disabled"}
                      />
                    </TableCell>
                    <TableCell>
                      <code>{w.secret_hint ?? "—"}</code>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1}>
                        <Button size="small" onClick={() => void openDeliveries(w.id)}>
                          View deliveries
                        </Button>
                        <Button size="small" color="error" onClick={() => void remove(w.id)}>
                          Delete
                        </Button>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
                {deliveries && Object.keys(deliveries).map((webhookId) => {
                  const list = deliveries[webhookId] ?? [];
                  if (list.length === 0) return null;
                  return (
                    <TableRow key={`deliveries-${webhookId}`}>
                      <TableCell colSpan={6} sx={{ bgcolor: "action.hover" }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Deliveries for <code>{webhookId.slice(0, 8)}…</code>
                        </Typography>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Event</TableCell>
                              <TableCell>Delivered</TableCell>
                              <TableCell>Status</TableCell>
                              <TableCell align="right">Attempts</TableCell>
                              <TableCell align="right">Response</TableCell>
                              <TableCell>Action</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {list.map((d) => (
                              <TableRow key={d.id}>
                                <TableCell>{d.event}</TableCell>
                                <TableCell>
                                  {new Date(d.delivered_at).toLocaleString()}
                                </TableCell>
                                <TableCell>
                                  <Chip
                                    size="small"
                                    color={deliveryColor(d.status)}
                                    label={d.status}
                                  />
                                </TableCell>
                                <TableCell align="right">{d.attempts ?? "—"}</TableCell>
                                <TableCell align="right">{d.response_code ?? "—"}</TableCell>
                                <TableCell>
                                  <Button
                                    size="small"
                                    onClick={() => void replay(webhookId, d.id)}
                                    data-testid={`webhook-replay-${d.id}`}
                                  >
                                    Replay
                                  </Button>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableCell>
                    </TableRow>
                  );
                })}
                {!loading && webhooks.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No webhooks registered.
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
        <DialogTitle>Register webhook</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              required
              label="URL"
              placeholder="https://acme.example.com/greenlang/webhook"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <TextField
              required
              label="Events (comma separated)"
              value={events}
              onChange={(e) => setEvents(e.target.value)}
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => void create()}
            disabled={busy || !url.trim()}
            data-testid="webhooks-save"
          >
            {busy ? "Registering…" : "Register"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FactorsWebhookRegistry;
