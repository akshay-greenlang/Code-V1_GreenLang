/**
 * F7.4 / Track B-5 — Edition diff viewer.
 *
 * Wired to: GET /v1/admin/diff/{from_edition}/{to_edition}
 *
 * Side-by-side diff between two editions: which factors were added,
 * removed, or had their fields changed. Used by methodology leads to
 * approve a release candidate before it ships.
 */
import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Grid from "@mui/material/Grid";
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
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import {
  FactorsApiError,
  diffEditions,
  type DiffFieldChange,
  type EditionDiffResponse,
} from "../lib/api/factorsClient";

type DiffAxis = "factor_version" | "release_version";

function ChangeTypeChip({ type }: { type: DiffFieldChange["type"] }) {
  const map: Record<DiffFieldChange["type"], "success" | "warning" | "error"> = {
    added: "success",
    changed: "warning",
    removed: "error",
  };
  return <Chip label={type} color={map[type]} size="small" />;
}

export function FactorsDiffViewer() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [diff, setDiff] = useState<EditionDiffResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [axis, setAxis] = useState<DiffAxis>("factor_version");

  const run = async () => {
    if (!from.trim() || !to.trim()) {
      setError("Both editions are required.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const r = await diffEditions(from.trim(), to.trim());
      setDiff(r);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      setDiff(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Edition Diff Viewer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Compare two editions side-by-side: factor adds, removes, and field-level changes.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            <ToggleButtonGroup
              value={axis}
              exclusive
              onChange={(_e, v) => {
                if (v) setAxis(v as DiffAxis);
              }}
              size="small"
              aria-label="Diff axis"
            >
              <ToggleButton value="factor_version" aria-label="Compare factor_version (numeric changes)">
                factor_version (numeric)
              </ToggleButton>
              <ToggleButton value="release_version" aria-label="Compare release_version (edition membership)">
                release_version (membership)
              </ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary">
              Dual axis per W4-D: `factor_version` surfaces numeric-value
              drift; `release_version` surfaces edition-membership moves
              (add/remove of factor records across editions).
            </Typography>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
              <TextField
                fullWidth
                label="From edition"
                placeholder="e.g. v1.0.0"
                value={from}
                onChange={(e) => setFrom(e.target.value)}
              />
              <TextField
                fullWidth
                label="To edition"
                placeholder="e.g. v1.1.0"
                value={to}
                onChange={(e) => setTo(e.target.value)}
              />
              <Button
                variant="contained"
                onClick={() => void run()}
                disabled={loading || !from.trim() || !to.trim()}
              >
                {loading ? "Diffing…" : "Compute diff"}
              </Button>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} role="alert">
          {error}
        </Alert>
      )}

      {diff && (
        <>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid xs={12} md={4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline" color="text.secondary">
                    Added
                  </Typography>
                  <Typography variant="h3" color="success.main">
                    {diff.added_factors.length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline" color="text.secondary">
                    Removed
                  </Typography>
                  <Typography variant="h3" color="error.main">
                    {diff.removed_factors.length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="overline" color="text.secondary">
                    Changed
                  </Typography>
                  <Typography variant="h3" color="warning.main">
                    {diff.changed_factors.length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Grid container spacing={2}>
            <Grid xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Added in {diff.to_edition}
                  </Typography>
                  <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 360 }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Factor ID</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {diff.added_factors.map((id) => (
                          <TableRow key={id}>
                            <TableCell><code>{id}</code></TableCell>
                          </TableRow>
                        ))}
                        {diff.added_factors.length === 0 && (
                          <TableRow>
                            <TableCell align="center">
                              <Typography variant="caption" color="text.secondary">
                                No additions.
                              </Typography>
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
            <Grid xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Removed from {diff.from_edition}
                  </Typography>
                  <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 360 }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Factor ID</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {diff.removed_factors.map((id) => (
                          <TableRow key={id}>
                            <TableCell><code>{id}</code></TableCell>
                          </TableRow>
                        ))}
                        {diff.removed_factors.length === 0 && (
                          <TableRow>
                            <TableCell align="center">
                              <Typography variant="caption" color="text.secondary">
                                No removals.
                              </Typography>
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Changed factors
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Factor</TableCell>
                      <TableCell>Field</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>From {diff.from_edition}</TableCell>
                      <TableCell>To {diff.to_edition}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {diff.changed_factors.flatMap((cf) =>
                      cf.changes
                        .filter((c) => {
                          if (axis === "release_version") {
                            return c.field === "release_version" || c.field === "edition_id";
                          }
                          // factor_version axis: numeric / record-rev changes
                          return (
                            c.field === "factor_version" ||
                            c.field === "co2e_per_unit" ||
                            c.field === "co2_per_unit" ||
                            c.field === "ch4_per_unit" ||
                            c.field === "n2o_per_unit" ||
                            c.field === "content_hash" ||
                            c.field.startsWith("factor_")
                          );
                        })
                        .map((c, idx) => (
                          <TableRow key={`${cf.factor_id}-${c.field}-${idx}`}>
                            <TableCell><code>{cf.factor_id}</code></TableCell>
                            <TableCell><code>{c.field}</code></TableCell>
                            <TableCell><ChangeTypeChip type={c.type} /></TableCell>
                            <TableCell>
                              <code>{JSON.stringify(c.old_value ?? "")}</code>
                            </TableCell>
                            <TableCell>
                              <code>{JSON.stringify(c.new_value ?? "")}</code>
                            </TableCell>
                          </TableRow>
                        )),
                    )}
                    {diff.changed_factors.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          <Typography variant="caption" color="text.secondary">
                            No field-level changes.
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
}

export default FactorsDiffViewer;
