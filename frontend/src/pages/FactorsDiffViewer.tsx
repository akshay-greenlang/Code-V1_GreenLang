/**
 * F7.4 — Factor diff viewer.
 *
 * Backing: GET /api/v1/factors/{factor_id}/diff?left=<edition>&right=<edition>
 * Renders a field-level diff (old_value / new_value) so operators can
 * see what changed between editions before approving a release.
 */
import { useState } from "react";
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

interface FieldChange {
  field: string;
  type: "added" | "removed" | "changed";
  old_value?: unknown;
  new_value?: unknown;
}

interface DiffResponse {
  factor_id: string;
  left_edition: string;
  right_edition: string;
  status: "unchanged" | "changed" | "added" | "removed" | "not_found";
  changes: FieldChange[];
  left_content_hash?: string;
  right_content_hash?: string;
}

export function FactorsDiffViewer() {
  const [factorId, setFactorId] = useState("");
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [diff, setDiff] = useState<DiffResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ left, right });
      const res = await fetch(
        `/api/v1/factors/${encodeURIComponent(factorId)}/diff?${qs}`,
      );
      if (!res.ok) throw new Error(`diff ${res.status}`);
      setDiff((await res.json()) as DiffResponse);
    } catch (e) {
      setError((e as Error).message);
      setDiff(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1100, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Factor Diff Viewer</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Compare a factor's field-by-field state across two editions.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <TextField
              fullWidth
              label="Factor ID"
              value={factorId}
              onChange={(e) => setFactorId(e.target.value)}
            />
            <TextField
              label="Left edition"
              value={left}
              onChange={(e) => setLeft(e.target.value)}
            />
            <TextField
              label="Right edition"
              value={right}
              onChange={(e) => setRight(e.target.value)}
            />
            <Button variant="contained" onClick={() => void run()} disabled={!factorId || !left || !right}>
              Diff
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {diff && (
        <>
          <Alert
            severity={
              diff.status === "unchanged"
                ? "success"
                : diff.status === "changed"
                  ? "warning"
                  : "info"
            }
            sx={{ mb: 2 }}
          >
            Status: <strong>{diff.status}</strong> — {diff.changes.length} field change(s).
          </Alert>

          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Field</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Old value</TableCell>
                  <TableCell>New value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {diff.changes.map((c) => (
                  <TableRow key={c.field}>
                    <TableCell><code>{c.field}</code></TableCell>
                    <TableCell>
                      <Chip
                        label={c.type}
                        color={c.type === "changed" ? "warning" : c.type === "added" ? "success" : "error"}
                        size="small"
                      />
                    </TableCell>
                    <TableCell><code>{JSON.stringify(c.old_value ?? "")}</code></TableCell>
                    <TableCell><code>{JSON.stringify(c.new_value ?? "")}</code></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
}

export default FactorsDiffViewer;
