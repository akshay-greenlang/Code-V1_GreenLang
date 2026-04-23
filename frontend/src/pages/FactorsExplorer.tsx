/**
 * Track B-5 — Factors Explorer (operator-gated).
 *
 * Wired to:
 *   GET  /v1/factors          (search)
 *   GET  /v1/factors/{id}     (detail)
 *   GET  /v1/factors/{id}/explain  (full resolution path)
 *
 * Searchable table + detail panel showing the factor record AND the full
 * explain payload (resolution_path, FQS components, citations, notes).
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
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
  explainFactor,
  getFactor,
  searchFactors,
  type FactorExplain,
  type FactorSummary,
  type FactorTier,
} from "../lib/api/factorsClient";

function TierBadge({ status }: { status?: FactorTier }) {
  const normalized = (status ?? "certified").toLowerCase();
  const styleByStatus: Record<string, { label: string; color: "success" | "warning" | "default" | "error" }> = {
    certified: { label: "Certified", color: "success" },
    preview: { label: "Preview", color: "warning" },
    connector_only: { label: "Connector-only", color: "default" },
    deprecated: { label: "Deprecated", color: "error" },
  };
  const s = styleByStatus[normalized] ?? styleByStatus.certified;
  return <Chip label={s.label} color={s.color} size="small" variant="outlined" />;
}

export function FactorsExplorer() {
  const [query, setQuery] = useState("");
  const [geography, setGeography] = useState("");
  const [scope, setScope] = useState("");
  const [family, setFamily] = useState("");
  const [results, setResults] = useState<FactorSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<(FactorSummary & Record<string, unknown>) | null>(null);
  const [explain, setExplain] = useState<FactorExplain | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      const res = await searchFactors(query, {
        geography: geography || undefined,
        scope: scope || undefined,
        family: family || undefined,
        limit: 50,
      });
      setResults(res.factors ?? []);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Initial empty-state: an empty query returns the first page so the
  // operator sees something useful on first paint.
  useEffect(() => {
    void runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openDetail = async (factorId: string) => {
    setSelectedId(factorId);
    setDetail(null);
    setExplain(null);
    setDetailError(null);
    setDetailLoading(true);
    // Fetch detail and explain independently so that a missing /explain
    // endpoint doesn't blank the whole panel.
    const [detailRes, explainRes] = await Promise.allSettled([
      getFactor(factorId),
      explainFactor(factorId),
    ]);
    if (detailRes.status === "fulfilled") {
      setDetail(detailRes.value);
    } else {
      const err = detailRes.reason;
      setDetailError(err instanceof FactorsApiError ? err.userMessage : (err as Error).message);
    }
    if (explainRes.status === "fulfilled") {
      setExplain(explainRes.value);
    } else if (detailRes.status === "fulfilled") {
      // Detail loaded but explain failed — surface a soft warning.
      const err = explainRes.reason;
      setDetailError(
        `Detail loaded; /explain failed: ${err instanceof FactorsApiError ? err.userMessage : (err as Error).message}`,
      );
    }
    setDetailLoading(false);
  };

  const closeDetail = () => {
    setSelectedId(null);
    setDetail(null);
    setExplain(null);
    setDetailError(null);
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Factors Explorer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Search and inspect emission factors across all sources. Click a row to
        see the full provenance chain (`/explain`), license, and data quality
        scores.
      </Typography>

      <Card>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <TextField
              label="Search"
              placeholder="e.g. 'diesel combustion' or 'grid electricity US'"
              fullWidth
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runSearch();
              }}
            />
            <TextField
              label="Geography"
              value={geography}
              onChange={(e) => setGeography(e.target.value)}
              sx={{ minWidth: 140 }}
            />
            <TextField
              label="Scope"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              sx={{ minWidth: 140 }}
            />
            <TextField
              label="Family"
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              sx={{ minWidth: 160 }}
            />
            <Button variant="contained" onClick={() => void runSearch()} disabled={loading}>
              Search
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress sx={{ mt: 2 }} />}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }} role="alert">
          {error}
        </Alert>
      )}

      <TableContainer component={Paper} sx={{ mt: 3 }} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Factor ID</TableCell>
              <TableCell>Family / fuel</TableCell>
              <TableCell>Geography</TableCell>
              <TableCell>Scope</TableCell>
              <TableCell align="right">CO₂e / unit</TableCell>
              <TableCell>Source</TableCell>
              <TableCell align="right">FQS</TableCell>
              <TableCell>Label</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {results.map((row) => (
              <TableRow
                key={row.factor_id}
                hover
                onClick={() => void openDetail(row.factor_id)}
                sx={{ cursor: "pointer" }}
              >
                <TableCell><code>{row.factor_id}</code></TableCell>
                <TableCell>{row.family ?? row.fuel_type ?? "—"}</TableCell>
                <TableCell>{row.geography ?? "—"}</TableCell>
                <TableCell>{row.scope ?? "—"}</TableCell>
                <TableCell align="right">
                  {row.co2e_per_unit?.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? "—"}
                  {row.unit ? ` / ${row.unit}` : ""}
                </TableCell>
                <TableCell>
                  {row.source ?? "—"}
                  {row.source_year ? ` (${row.source_year})` : ""}
                </TableCell>
                <TableCell align="right">
                  {(row.fqs ?? row.data_quality_score)?.toFixed?.(1) ?? "—"}
                </TableCell>
                <TableCell>
                  <TierBadge status={row.factor_status} />
                </TableCell>
              </TableRow>
            ))}
            {!loading && hasSearched && results.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No results. Try searching for e.g. "diesel" or "electricity".
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={selectedId !== null} onClose={closeDetail} maxWidth="md" fullWidth>
        <DialogTitle>
          Factor detail
          {selectedId && (
            <Typography variant="caption" display="block">
              <code>{selectedId}</code>
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading && <LinearProgress />}
          {detailError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {detailError}
            </Alert>
          )}
          {detail && (
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle2">Summary</Typography>
                <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
                  <TierBadge status={detail.factor_status} />
                  {detail.license_class && (
                    <Chip size="small" label={`License: ${detail.license_class}`} />
                  )}
                  {detail.geography && <Chip size="small" label={`Geo: ${detail.geography}`} />}
                  {detail.scope && <Chip size="small" label={`Scope: ${detail.scope}`} />}
                </Stack>
              </Box>

              <Divider />

              <Box>
                <Typography variant="subtitle2">Resolution path (`/explain`)</Typography>
                {explain ? (
                  <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Step</TableCell>
                          <TableCell>Source</TableCell>
                          <TableCell>Score</TableCell>
                          <TableCell>Rationale</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(explain.resolution_path ?? []).map((p, idx) => (
                          <TableRow key={`${p.step}-${idx}`}>
                            <TableCell><code>{p.step}</code></TableCell>
                            <TableCell>{p.source ?? "—"}</TableCell>
                            <TableCell>{p.score?.toFixed?.(2) ?? "—"}</TableCell>
                            <TableCell>{p.rationale ?? "—"}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    No explain payload available for this factor.
                  </Typography>
                )}
              </Box>

              {explain?.components && (
                <Box>
                  <Typography variant="subtitle2">FQS components</Typography>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
                    {Object.entries(explain.components).map(([k, v]) => (
                      <Chip key={k} size="small" variant="outlined" label={`${k}: ${v.toFixed(1)}`} />
                    ))}
                  </Stack>
                </Box>
              )}

              {explain?.citations && explain.citations.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">Citations</Typography>
                  <Stack spacing={0.5} sx={{ mt: 1 }}>
                    {explain.citations.map((c) => (
                      <Typography key={c.id} variant="body2">
                        <code>{c.id}</code> — {c.title}
                        {c.url && (
                          <>
                            {" "}
                            <a href={c.url} target="_blank" rel="noreferrer">
                              link
                            </a>
                          </>
                        )}
                      </Typography>
                    ))}
                  </Stack>
                </Box>
              )}

              <Divider />

              <Box>
                <Typography variant="subtitle2">Raw factor payload</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", maxHeight: 280, overflow: "auto", fontSize: 12 }}>
                  {JSON.stringify(detail, null, 2)}
                </pre>
              </Box>
            </Stack>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}

export default FactorsExplorer;
