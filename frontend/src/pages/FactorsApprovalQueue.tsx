/**
 * F7.5 — Approval queue.
 *
 * Lists factors in `preview` status awaiting methodology-lead sign-off
 * before promotion to `certified`. Backing:
 *   GET /api/v1/factors/quality/review_queue
 *   POST /api/v1/factors/quality/review/{id}/decide
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
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
import Typography from "@mui/material/Typography";

interface ReviewItem {
  review_id: string;
  factor_id: string;
  current_status: string;
  proposed_status: string;
  submitted_by: string;
  submitted_at: string;
  rationale: string;
  reviewer: string | null;
  due_date: string | null;
}

export function FactorsApprovalQueue() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/factors/quality/review_queue");
      if (!res.ok) throw new Error(`status ${res.status}`);
      const payload = await res.json();
      setItems((payload.items ?? []) as ReviewItem[]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const decide = async (id: string, decision: "approve" | "reject" | "needs_revision") => {
    try {
      const res = await fetch(
        `/api/v1/factors/quality/review/${encodeURIComponent(id)}/decide`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decision }),
        },
      );
      if (!res.ok) throw new Error(`decide ${res.status}`);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Approval Queue</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Factors awaiting methodology-lead sign-off. Approving promotes to
        <code> certified</code>; rejection or "needs revision" sends back
        with notes.
      </Typography>

      {loading && <LinearProgress />}
      {error && <Alert severity="info" sx={{ mb: 2 }}>{error} — endpoint may not be wired yet.</Alert>}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Review ID</TableCell>
              <TableCell>Factor</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Submitted by</TableCell>
              <TableCell>Rationale</TableCell>
              <TableCell>Reviewer</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {items.map((it) => (
              <TableRow key={it.review_id}>
                <TableCell><code>{it.review_id.slice(0, 10)}…</code></TableCell>
                <TableCell><code>{it.factor_id}</code></TableCell>
                <TableCell>
                  <Chip label={it.current_status} size="small" />
                  {"→"}
                  <Chip label={it.proposed_status} color="primary" size="small" />
                </TableCell>
                <TableCell>{it.submitted_by}</TableCell>
                <TableCell>{it.rationale}</TableCell>
                <TableCell>{it.reviewer ?? "unassigned"}</TableCell>
                <TableCell>
                  <Stack direction="row" spacing={1}>
                    <Button size="small" color="success" onClick={() => void decide(it.review_id, "approve")}>
                      Approve
                    </Button>
                    <Button size="small" color="warning" onClick={() => void decide(it.review_id, "needs_revision")}>
                      Revise
                    </Button>
                    <Button size="small" color="error" onClick={() => void decide(it.review_id, "reject")}>
                      Reject
                    </Button>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
            {!loading && items.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    Review queue is empty.
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

export default FactorsApprovalQueue;
