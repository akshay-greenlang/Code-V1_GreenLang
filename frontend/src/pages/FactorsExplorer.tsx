/**
 * Phase 5.1 — Factor Explorer UI.
 *
 * Search, filter, and inspect factors against the hosted Factors API.
 * Wires to:
 *   - GET /api/v1/factors/search          (full-text)
 *   - GET /api/v1/factors/search/facets   (filter choices)
 *   - GET /api/v1/factors/{factor_id}     (detail modal)
 *
 * Requires JWT or API-Key authentication (the middleware enforces it).
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import FormControl from "@mui/material/FormControl";
import IconButton from "@mui/material/IconButton";
import InputLabel from "@mui/material/InputLabel";
import LinearProgress from "@mui/material/LinearProgress";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

interface FactorSummary {
  factor_id: string;
  fuel_type: string;
  unit: string;
  geography: string;
  scope: string;
  co2e_per_unit: number;
  source: string;
  source_year: number;
  data_quality_score: number;
  factor_status: string;
  source_id?: string | null;
  license_class?: string | null;
}

interface SearchResponse {
  factors: FactorSummary[];
  total_count?: number;
  edition_id?: string;
}

interface FacetsResponse {
  edition_id: string;
  facets: Record<string, Record<string, number>>;
}

function TierBadge({ status }: { status: string }) {
  const normalized = (status || "certified").toLowerCase();
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
  const [geography, setGeography] = useState<string>("");
  const [scope, setScope] = useState<string>("");
  const [fuelType, setFuelType] = useState<string>("");
  const [results, setResults] = useState<FactorSummary[]>([]);
  const [facets, setFacets] = useState<FacetsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Preload filter choices.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/v1/factors/search/facets")
      .then(async (r) => {
        if (!r.ok) throw new Error(`Facets ${r.status}`);
        return r.json();
      })
      .then((data: FacetsResponse) => {
        if (!cancelled) setFacets(data);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(`Could not load filters: ${e.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("q", query || "*");
      if (geography) params.set("geography", geography);
      params.set("limit", "50");
      const res = await fetch(`/api/v1/factors/search?${params.toString()}`);
      if (!res.ok) throw new Error(`Search ${res.status}`);
      const payload = (await res.json()) as SearchResponse;
      let rows = payload.factors ?? [];
      // Client-side fuel_type + scope refinement (server's POST /search/v2
      // supports these natively; we keep GET /search simple here).
      if (fuelType) rows = rows.filter((r) => r.fuel_type?.toLowerCase() === fuelType.toLowerCase());
      if (scope) rows = rows.filter((r) => r.scope === scope);
      setResults(rows);
    } catch (e) {
      setError((e as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const openDetail = async (factorId: string) => {
    setSelected(factorId);
    setDetail(null);
    setDetailLoading(true);
    try {
      const res = await fetch(`/api/v1/factors/${encodeURIComponent(factorId)}`);
      if (!res.ok) throw new Error(`Detail ${res.status}`);
      setDetail((await res.json()) as Record<string, unknown>);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDetailLoading(false);
    }
  };

  const closeDetail = () => {
    setSelected(null);
    setDetail(null);
  };

  const geographyChoices = useMemo(
    () => Object.keys(facets?.facets["geography"] ?? {}).sort(),
    [facets],
  );
  const scopeChoices = useMemo(
    () => Object.keys(facets?.facets["scope"] ?? {}).sort(),
    [facets],
  );
  const fuelChoices = useMemo(
    () => Object.keys(facets?.facets["fuel_type"] ?? {}).sort(),
    [facets],
  );

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        GreenLang Factors — Explorer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Search and inspect emission factors across all sources. Click a row to
        see the full provenance chain, license, and data quality scores.
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
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="geo-label">Geography</InputLabel>
              <Select
                labelId="geo-label"
                label="Geography"
                value={geography}
                onChange={(e) => setGeography(e.target.value)}
              >
                <MenuItem value="">Any</MenuItem>
                {geographyChoices.map((g) => (
                  <MenuItem key={g} value={g}>
                    {g}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 140 }}>
              <InputLabel id="scope-label">Scope</InputLabel>
              <Select
                labelId="scope-label"
                label="Scope"
                value={scope}
                onChange={(e) => setScope(e.target.value)}
              >
                <MenuItem value="">Any</MenuItem>
                {scopeChoices.map((s) => (
                  <MenuItem key={s} value={s}>
                    {s}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 180 }}>
              <InputLabel id="fuel-label">Fuel / activity</InputLabel>
              <Select
                labelId="fuel-label"
                label="Fuel / activity"
                value={fuelType}
                onChange={(e) => setFuelType(e.target.value)}
              >
                <MenuItem value="">Any</MenuItem>
                {fuelChoices.map((f) => (
                  <MenuItem key={f} value={f}>
                    {f}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button variant="contained" onClick={() => void runSearch()} disabled={loading}>
              Search
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {loading && <LinearProgress sx={{ mt: 2 }} />}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper} sx={{ mt: 3 }} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Factor ID</TableCell>
              <TableCell>Fuel / activity</TableCell>
              <TableCell>Geography</TableCell>
              <TableCell>Scope</TableCell>
              <TableCell align="right">CO₂e / unit</TableCell>
              <TableCell>Source</TableCell>
              <TableCell align="right">DQS</TableCell>
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
                <TableCell>
                  <code>{row.factor_id}</code>
                </TableCell>
                <TableCell>{row.fuel_type}</TableCell>
                <TableCell>{row.geography}</TableCell>
                <TableCell>{row.scope}</TableCell>
                <TableCell align="right">
                  {row.co2e_per_unit?.toLocaleString(undefined, {
                    maximumFractionDigits: 4,
                  })}{" "}
                  / {row.unit}
                </TableCell>
                <TableCell>
                  {row.source} ({row.source_year})
                </TableCell>
                <TableCell align="right">{row.data_quality_score?.toFixed(1)}</TableCell>
                <TableCell>
                  <TierBadge status={row.factor_status} />
                </TableCell>
              </TableRow>
            ))}
            {!loading && results.length === 0 && (
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

      <Dialog open={selected !== null} onClose={closeDetail} maxWidth="md" fullWidth>
        <DialogTitle>
          Factor Detail
          {selected && (
            <Typography variant="caption" display="block">
              <code>{selected}</code>
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading && <LinearProgress />}
          {detail && (
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle2">Source</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(detail["source"] ?? {}, null, 2)}
                </pre>
              </Box>
              <Box>
                <Typography variant="subtitle2">Data Quality Score</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(detail["data_quality"] ?? {}, null, 2)}
                </pre>
              </Box>
              <Box>
                <Typography variant="subtitle2">License</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(
                    {
                      license: detail["license"],
                      license_class: detail["license_class"],
                      redistribution_allowed: detail["redistribution_allowed"],
                    },
                    null,
                    2,
                  )}
                </pre>
              </Box>
              <Box>
                <Typography variant="subtitle2">Compliance frameworks</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(detail["compliance_frameworks"] ?? [], null, 2)}
                </pre>
              </Box>
              <Box>
                <Typography variant="subtitle2">Raw factor payload</Typography>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", maxHeight: 300, overflow: "auto" }}>
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
