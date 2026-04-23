/**
 * F7.5 / Track B-5 — Methodology lead's approval queue.
 *
 * Wired to:
 *   GET  /v1/admin/queue                    — open review items
 *   POST /v1/admin/queue/{id}/approve       — promote to next status
 *   POST /v1/admin/queue/{id}/reject        — reject with reason
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
  approve,
  getQueue,
  reject,
  type QueueItem,
} from "../lib/api/factorsClient";

export function FactorsApprovalQueue() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [rejectTarget, setRejectTarget] = useState<QueueItem | null>(null);
  const [rejectReason, setRejectReason] = useState("");
  const [evidenceTarget, setEvidenceTarget] = useState<QueueItem | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const r = await getQueue();
      setItems(r.items ?? []);
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

  const onApprove = async (item: QueueItem) => {
    setBusy(item.review_id);
    try {
      await approve(item.review_id);
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const submitReject = async () => {
    if (!rejectTarget) return;
    setBusy(rejectTarget.review_id);
    try {
      await reject(rejectTarget.review_id, rejectReason || "rejected");
      setRejectTarget(null);
      setRejectReason("");
      await load();
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Approval Queue
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Methodology lead's review queue. Approving promotes a factor to its proposed status (most
        commonly <code>preview</code> → <code>certified</code>); rejection sends back with a
        reason.
      </Typography>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Review ID</TableCell>
                  <TableCell>Factor / family</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Submitted by</TableCell>
                  <TableCell>Submitted at</TableCell>
                  <TableCell>Rationale</TableCell>
                  <TableCell>Reviewer</TableCell>
                  <TableCell>Evidence</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {items.map((it) => (
                  <TableRow key={it.review_id}>
                    <TableCell><code>{it.review_id.slice(0, 10)}…</code></TableCell>
                    <TableCell>
                      {it.factor_id && <code>{it.factor_id}</code>}
                      {it.family && (
                        <Typography variant="caption" display="block" color="text.secondary">
                          family: {it.family}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} alignItems="center">
                        <Chip label={it.current_status} size="small" />
                        <span>→</span>
                        <Chip label={it.proposed_status} color="primary" size="small" />
                      </Stack>
                    </TableCell>
                    <TableCell>{it.submitted_by}</TableCell>
                    <TableCell>{new Date(it.submitted_at).toLocaleString()}</TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 280 }}>
                        {it.rationale}
                      </Typography>
                    </TableCell>
                    <TableCell>{it.reviewer ?? "unassigned"}</TableCell>
                    <TableCell>
                      {it.evidence ? (
                        <Button size="small" onClick={() => setEvidenceTarget(it)}>
                          View
                        </Button>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1}>
                        <Button
                          size="small"
                          color="success"
                          variant="outlined"
                          onClick={() => void onApprove(it)}
                          disabled={busy === it.review_id}
                        >
                          Approve
                        </Button>
                        <Button
                          size="small"
                          color="error"
                          variant="outlined"
                          onClick={() => {
                            setRejectTarget(it);
                            setRejectReason("");
                          }}
                          disabled={busy === it.review_id}
                        >
                          Reject
                        </Button>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
                {!loading && items.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={9} align="center">
                      <Typography variant="body2" color="text.secondary">
                        Review queue is empty.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Dialog open={rejectTarget !== null} onClose={() => setRejectTarget(null)}>
        <DialogTitle>Reject review</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Provide a reason. The submitter will see this on their review.
          </Typography>
          <TextField
            autoFocus
            fullWidth
            multiline
            minRows={3}
            label="Reason"
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRejectTarget(null)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={() => void submitReject()} disabled={!rejectReason.trim()}>
            Reject
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={evidenceTarget !== null} onClose={() => setEvidenceTarget(null)} maxWidth="md" fullWidth>
        <DialogTitle>Evidence</DialogTitle>
        <DialogContent>
          <pre style={{ margin: 0, fontSize: 12, whiteSpace: "pre-wrap", maxHeight: 480, overflow: "auto" }}>
            {JSON.stringify(evidenceTarget?.evidence ?? {}, null, 2)}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEvidenceTarget(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FactorsApprovalQueue;
