/**
 * O5 / W4-D — Method-pack manager.
 *
 * Wired to GET /v1/method-packs (and /v1/admin/method-packs for status
 * mutation; mutation surface is read-only in v1 — operators file a
 * deprecation via the approval queue).
 *
 * Surfaces per pack:
 *   - version / status / deprecation
 *   - cannot_resolve_safely error counts
 *   - conformance test pass/total ratio
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
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
import Typography from "@mui/material/Typography";
import {
  FactorsApiError,
  listMethodPacks,
  type MethodPackSummary,
} from "../lib/api/factorsClient";

function statusColor(status?: string): "success" | "warning" | "error" | "default" {
  const s = (status ?? "").toLowerCase();
  if (s === "active") return "success";
  if (s === "draft") return "default";
  if (s === "deprecated") return "warning";
  if (s === "retired") return "error";
  return "default";
}

export function FactorsMethodPackManager() {
  const [packs, setPacks] = useState<MethodPackSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const r = await listMethodPacks();
        if (!cancelled) {
          setPacks(r.method_packs ?? []);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1300, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Method-Pack Manager
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Per-pack version, status, deprecation schedule, unsafe-resolve
        counts, and conformance test coverage.
      </Typography>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} role="alert">
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small" aria-label="Method packs">
              <TableHead>
                <TableRow>
                  <TableCell>Pack</TableCell>
                  <TableCell>Version</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Scope</TableCell>
                  <TableCell>Jurisdictions</TableCell>
                  <TableCell align="right">cannot_resolve_safely</TableCell>
                  <TableCell align="right">Conformance</TableCell>
                  <TableCell>Replacement</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {packs.map((p) => (
                  <TableRow key={p.method_pack_id} hover>
                    <TableCell>
                      <code>{p.method_pack_id}</code>
                      {p.name && (
                        <Typography variant="caption" display="block" color="text.secondary">
                          {p.name}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>{p.version ?? "—"}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={p.status ?? "unknown"}
                        color={statusColor(p.status)}
                        data-testid={`method-pack-status-${p.method_pack_id}`}
                      />
                      {p.deprecated_at && (
                        <Typography variant="caption" display="block" color="warning.main">
                          deprecated {new Date(p.deprecated_at).toLocaleDateString()}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>{p.scope ?? "—"}</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {(p.jurisdictions ?? []).slice(0, 4).map((j) => (
                          <Chip key={j} size="small" variant="outlined" label={j} />
                        ))}
                        {(p.jurisdictions ?? []).length > 4 && (
                          <Chip
                            size="small"
                            variant="outlined"
                            label={`+${(p.jurisdictions ?? []).length - 4}`}
                          />
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell align="right">
                      <Chip
                        size="small"
                        color={(p.cannot_resolve_safely_count ?? 0) > 0 ? "warning" : "default"}
                        label={(p.cannot_resolve_safely_count ?? 0).toLocaleString()}
                      />
                    </TableCell>
                    <TableCell align="right">
                      {p.conformance_total ? (
                        <Chip
                          size="small"
                          color={
                            p.conformance_pass === p.conformance_total
                              ? "success"
                              : p.conformance_pass! / p.conformance_total < 0.95
                                ? "error"
                                : "warning"
                          }
                          label={`${p.conformance_pass ?? 0}/${p.conformance_total}`}
                        />
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell>
                      {p.replacement_pack_id ? <code>{p.replacement_pack_id}</code> : "—"}
                    </TableCell>
                  </TableRow>
                ))}
                {!loading && packs.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No method packs registered.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}

export default FactorsMethodPackManager;
