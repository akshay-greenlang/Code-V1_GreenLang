/**
 * Track B-5 / W4-D — Factors Explorer rewired to v1.2.0 SDK shape.
 *
 * Detail panel surfaces the full Wave 2/2a/2.5 envelope:
 *   1. `chosen_factor` ........ header chips
 *   2. `source` (SourceDescriptor) identification
 *   3. `quality` (composite_fqs_0_100) → FQSGauge
 *   4. `uncertainty` (ci_95, distribution, pedigree) → ±band
 *   5. `licensing` envelope → LicenseClassBadge
 *   6. `deprecation_status` → DeprecationBanner
 *   7. `gas_breakdown` → GasBreakdownTable
 *   8. `audit_text` + `audit_text_draft` → AuditTextPanel
 *   9. `signed_receipt` (top-level Wave 2a) → collapsible debug pane
 *  10-16. factor_id, factor_version, release_version, method_profile,
 *         method_pack_id, co2e_per_unit + unit, geography/scope (16 fields).
 *
 * Pages without a resolve are rendered from the factor summary; clicking
 * a row triggers a live `/resolve-explain` call so the operator sees the
 * v1.2.0 envelope end-to-end.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Collapse from "@mui/material/Collapse";
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
import { AuditTextPanel } from "../components/AuditTextPanel";
import { DeprecationBanner } from "../components/DeprecationBanner";
import { FQSGauge } from "../components/FQSGauge";
import { GasBreakdownTable } from "../components/GasBreakdownTable";
import { LicenseClassBadge } from "../components/LicenseClassBadge";
import {
  formatApiError,
  safeResolve,
  type ResolvedFactor,
} from "../lib/factorsClient";
import {
  FactorsApiError,
  searchFactors,
  type FactorSummary,
} from "../lib/api/factorsClient";

function HeaderChips({ resolved }: { resolved: ResolvedFactor }) {
  const chosen = resolved.chosen_factor;
  const chips: Array<{ label: string }> = [];
  if (chosen?.factor_id) chips.push({ label: `factor_id: ${chosen.factor_id}` });
  if (chosen?.factor_version) chips.push({ label: `factor_version: ${chosen.factor_version}` });
  if (chosen?.release_version) chips.push({ label: `release_version: ${chosen.release_version}` });
  if (chosen?.method_profile) chips.push({ label: `method_profile: ${chosen.method_profile}` });
  if (chosen?.method_pack_id) chips.push({ label: `pack: ${chosen.method_pack_id}` });
  if (chosen?.geography) chips.push({ label: `geo: ${chosen.geography}` });
  if (chosen?.scope) chips.push({ label: `scope: ${chosen.scope}` });
  if (chosen?.co2e_per_unit !== undefined && chosen?.co2e_per_unit !== null) {
    chips.push({
      label: `${chosen.co2e_per_unit.toLocaleString(undefined, {
        maximumFractionDigits: 6,
      })} ${chosen.unit ?? ""}`.trim(),
    });
  }
  return (
    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 0.5 }}>
      {chips.map((c) => (
        <Chip key={c.label} label={c.label} size="small" variant="outlined" />
      ))}
    </Stack>
  );
}

function UncertaintyBand({ resolved }: { resolved: ResolvedFactor }) {
  const u = resolved.uncertainty;
  if (!u) {
    return (
      <Typography variant="caption" color="text.secondary">
        No uncertainty envelope surfaced.
      </Typography>
    );
  }
  const mid = resolved.chosen_factor?.co2e_per_unit;
  const ci95 = typeof u.ci_95 === "number" ? u.ci_95 : undefined;
  return (
    <Stack spacing={0.5}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        {ci95 !== undefined && (
          <Chip size="small" label={`±${ci95.toFixed(2)}% (95% CI)`} color="info" />
        )}
        {typeof u.ci_lower === "number" && typeof u.ci_upper === "number" && (
          <Chip
            size="small"
            variant="outlined"
            label={`[${u.ci_lower}, ${u.ci_upper}]`}
          />
        )}
        {u.distribution && <Chip size="small" variant="outlined" label={`dist: ${u.distribution}`} />}
        {typeof u.sample_size === "number" && (
          <Chip size="small" variant="outlined" label={`n=${u.sample_size}`} />
        )}
      </Stack>
      {mid !== undefined && mid !== null && ci95 !== undefined && (
        <Typography variant="caption" color="text.secondary">
          central value {mid.toFixed(6)} ± {((mid * ci95) / 100).toFixed(6)} @ 95% CI
        </Typography>
      )}
    </Stack>
  );
}

function SignedReceiptPane({ resolved }: { resolved: ResolvedFactor }) {
  const [open, setOpen] = useState(false);
  const r = resolved.signed_receipt;
  if (!r) {
    return (
      <Typography variant="caption" color="text.secondary">
        No signed receipt attached.
      </Typography>
    );
  }
  return (
    <Stack spacing={1} data-testid="signed-receipt-pane">
      <Button
        size="small"
        variant="outlined"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="signed-receipt-body"
      >
        {open ? "Hide" : "Show"} signed receipt debug
      </Button>
      <Collapse in={open} unmountOnExit>
        <Paper
          variant="outlined"
          id="signed-receipt-body"
          sx={{ p: 1.5, bgcolor: "action.hover" }}
        >
          <Stack spacing={0.5}>
            <Typography variant="caption">
              <strong>receipt_id:</strong> <code>{r.receipt_id ?? "—"}</code>
            </Typography>
            <Typography variant="caption">
              <strong>alg:</strong> <code>{r.alg}</code>
            </Typography>
            <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
              <strong>payload_hash:</strong> <code>{r.payload_hash ?? "—"}</code>
            </Typography>
            <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
              <strong>signature:</strong> <code>{r.signature.slice(0, 64)}…</code>
            </Typography>
            <Typography variant="caption">
              <strong>verification_key_hint:</strong>{" "}
              <code>{r.verification_key_hint ?? "—"}</code>
            </Typography>
            <Typography variant="caption">
              <strong>signed_at:</strong> {r.signed_at ?? "—"}
            </Typography>
            <Typography variant="caption">
              <strong>edition_id:</strong> {r.edition_id ?? "—"}
            </Typography>
          </Stack>
        </Paper>
      </Collapse>
    </Stack>
  );
}

export function FactorsExplorer() {
  // Search state
  const [query, setQuery] = useState("");
  const [geography, setGeography] = useState("");
  const [scope, setScope] = useState("");
  const [family, setFamily] = useState("");
  const [results, setResults] = useState<FactorSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  // Detail state (Wave 2 envelope)
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [resolved, setResolved] = useState<ResolvedFactor | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [resolveModalOpen, setResolveModalOpen] = useState(false);
  const [resolveModalError, setResolveModalError] = useState<string | null>(null);

  const runSearch = async () => {
    setLoading(true);
    setSearchError(null);
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
      setSearchError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openDetail = async (row: FactorSummary) => {
    setSelectedId(row.factor_id);
    setResolved(null);
    setDetailError(null);
    setDetailLoading(true);
    const result = await safeResolve({
      activity: row.factor_id,
      method_profile: "default",
      jurisdiction: row.geography,
    });
    if (result.ok) {
      setResolved(result.resolved);
    } else if (result.error.code === "factor_cannot_resolve_safely") {
      // Show the helpful modal in addition to the inline error.
      setResolveModalError(result.error.message);
      setResolveModalOpen(true);
      setDetailError(result.error.message);
    } else {
      setDetailError(result.error.message);
    }
    setDetailLoading(false);
  };

  const closeDetail = () => {
    setSelectedId(null);
    setResolved(null);
    setDetailError(null);
  };

  const fqs = useMemo(() => {
    if (!resolved?.quality) return null;
    return (
      resolved.quality.composite_fqs_0_100 ??
      (typeof resolved.quality.overall_score === "number"
        ? resolved.quality.overall_score * 10
        : null)
    );
  }, [resolved]);

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1400, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Factors Explorer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Search the full catalog and inspect the v1.2.0 Wave 2 envelope:
        chosen_factor, composite FQS (0-100), uncertainty, licensing,
        deprecation status, gas breakdown, audit narrative (+ draft
        banner), and the top-level signed_receipt.
      </Typography>

      <Card>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <TextField
              label="Search"
              placeholder="e.g. 'grid electricity India' or 'diesel combustion'"
              fullWidth
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runSearch();
              }}
              inputProps={{ "aria-label": "Search factor catalog" }}
            />
            <TextField
              label="Geography"
              value={geography}
              onChange={(e) => setGeography(e.target.value)}
              sx={{ minWidth: 140 }}
              inputProps={{ "aria-label": "Filter by geography" }}
            />
            <TextField
              label="Scope"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              sx={{ minWidth: 140 }}
              inputProps={{ "aria-label": "Filter by GHG Protocol scope" }}
            />
            <TextField
              label="Family"
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              sx={{ minWidth: 160 }}
              inputProps={{ "aria-label": "Filter by factor family" }}
            />
            <Button
              variant="contained"
              onClick={() => void runSearch()}
              disabled={loading}
              aria-label="Run search"
            >
              Search
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress sx={{ mt: 2 }} />}
      {searchError && (
        <Alert severity="error" sx={{ mt: 2 }} role="alert">
          {searchError}
        </Alert>
      )}

      <TableContainer component={Paper} sx={{ mt: 3 }} variant="outlined">
        <Table size="small" aria-label="Factor search results">
          <TableHead>
            <TableRow>
              <TableCell>Factor ID</TableCell>
              <TableCell>Family / fuel</TableCell>
              <TableCell>Geography</TableCell>
              <TableCell>Scope</TableCell>
              <TableCell align="right">CO₂e / unit</TableCell>
              <TableCell>License</TableCell>
              <TableCell>FQS</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {results.map((row) => (
              <TableRow
                key={row.factor_id}
                hover
                tabIndex={0}
                onClick={() => void openDetail(row)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void openDetail(row);
                }}
                sx={{ cursor: "pointer" }}
                aria-label={`Open resolve detail for ${row.factor_id}`}
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
                  <LicenseClassBadge cls={row.license_class ?? row.factor_status} noTooltip />
                </TableCell>
                <TableCell>
                  <FQSGauge score={row.fqs ?? row.data_quality_score} compact />
                </TableCell>
              </TableRow>
            ))}
            {!loading && hasSearched && results.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
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
          Wave 2 envelope
          {selectedId && (
            <Typography variant="caption" display="block">
              <code>{selectedId}</code>
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers data-testid="factor-detail-panel">
          {detailLoading && <LinearProgress />}
          {detailError && (
            <Alert severity="error" sx={{ mb: 2 }} role="alert">
              {detailError}
            </Alert>
          )}
          {resolved && (
            <Stack spacing={2}>
              {/* Field 1: chosen_factor */}
              <Box data-testid="section-chosen-factor">
                <Typography variant="subtitle2">Chosen factor</Typography>
                <Box sx={{ mt: 1 }}>
                  <HeaderChips resolved={resolved} />
                </Box>
              </Box>

              {/* Field 6: deprecation banner (renders only when not active) */}
              <DeprecationBanner status={resolved.deprecation_status} />

              <Divider />

              <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                {/* Field 3: quality envelope */}
                <Box sx={{ flex: 1 }} data-testid="section-quality">
                  <Typography variant="subtitle2">Composite FQS (0-100)</Typography>
                  <Box sx={{ mt: 1 }}>
                    <FQSGauge score={fqs} />
                  </Box>
                </Box>
                {/* Field 4: uncertainty envelope */}
                <Box sx={{ flex: 1 }} data-testid="section-uncertainty">
                  <Typography variant="subtitle2">Uncertainty (±band)</Typography>
                  <Box sx={{ mt: 1 }}>
                    <UncertaintyBand resolved={resolved} />
                  </Box>
                </Box>
              </Stack>

              <Divider />

              {/* Field 5: licensing envelope */}
              <Box data-testid="section-licensing">
                <Typography variant="subtitle2">Licensing</Typography>
                <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap", gap: 0.5 }}>
                  <LicenseClassBadge cls={resolved.licensing?.license_class} />
                  {resolved.licensing?.license && (
                    <Chip size="small" variant="outlined" label={`license: ${resolved.licensing.license}`} />
                  )}
                  {resolved.licensing?.redistribution_class && (
                    <Chip
                      size="small"
                      variant="outlined"
                      label={`redistribution: ${resolved.licensing.redistribution_class}`}
                    />
                  )}
                  {(resolved.licensing?.upstream_licenses ?? []).map((u) => (
                    <Chip key={u} size="small" label={`upstream: ${u}`} variant="outlined" />
                  ))}
                </Stack>
                {resolved.licensing?.attribution && (
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                    Attribution: {resolved.licensing.attribution}
                  </Typography>
                )}
              </Box>

              {/* Field 2: source descriptor */}
              <Box data-testid="section-source">
                <Typography variant="subtitle2">Source</Typography>
                {resolved.source ? (
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap", gap: 0.5 }}>
                    <Chip size="small" label={resolved.source.source_id} />
                    {resolved.source.organization && (
                      <Chip size="small" variant="outlined" label={resolved.source.organization} />
                    )}
                    {resolved.source.publication && (
                      <Chip size="small" variant="outlined" label={resolved.source.publication} />
                    )}
                    {resolved.source.year && (
                      <Chip size="small" variant="outlined" label={`${resolved.source.year}`} />
                    )}
                  </Stack>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    No source descriptor.
                  </Typography>
                )}
              </Box>

              <Divider />

              {/* Field 7: gas breakdown */}
              <Box data-testid="section-gas-breakdown">
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Gas breakdown
                </Typography>
                <GasBreakdownTable
                  gas={resolved.gas_breakdown}
                  unit={resolved.chosen_factor?.unit ?? undefined}
                />
              </Box>

              <Divider />

              {/* Field 8: audit_text + draft banner */}
              <Box data-testid="section-audit-text">
                <AuditTextPanel
                  auditText={resolved.audit_text}
                  draft={resolved.audit_text_draft}
                />
              </Box>

              <Divider />

              {/* Field 9: signed_receipt debug pane */}
              <Box data-testid="section-signed-receipt">
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Signed receipt
                </Typography>
                <SignedReceiptPane resolved={resolved} />
              </Box>
            </Stack>
          )}
        </DialogContent>
      </Dialog>

      {/* FactorCannotResolveSafelyError → helpful modal */}
      <Dialog open={resolveModalOpen} onClose={() => setResolveModalOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>No safe factor could be resolved</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            {resolveModalError ?? "No candidate met the method pack's safety floor."}
          </Alert>
          <Typography variant="body2" gutterBottom>
            Try one of:
          </Typography>
          <Stack spacing={0.5} sx={{ pl: 2 }}>
            <Typography variant="body2">• Loosen to a less strict method pack</Typography>
            <Typography variant="body2">• Upload a customer-specific factor override</Typography>
            <Typography variant="body2">• Accept preview factors (include_preview=true)</Typography>
            <Typography variant="body2">• Contact methodology@greenlang.io for an exception</Typography>
          </Stack>
        </DialogContent>
      </Dialog>
    </Box>
  );
}

export default FactorsExplorer;

// For internal/test access
export { formatApiError };
