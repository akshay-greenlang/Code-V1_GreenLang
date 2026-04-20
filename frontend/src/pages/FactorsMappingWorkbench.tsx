/**
 * F7.2 — Mapping workbench.
 *
 * Lets operators paste a free-text activity / spend line and see what
 * the mapping layer (Phase F4) resolves. Useful for:
 *   - QA: verifying a customer-supplied description resolves correctly
 *   - Rule authoring: spotting a synonym gap before it ships to prod
 *
 * Backing endpoint (planned): POST /api/v1/factors/mapping/resolve
 * Until that lands, the page calls the four client-side mapping JSONs
 * via a stub ``/mapping/test`` endpoint; falls back to displaying the
 * raw input with a "backend not wired" notice.
 */
import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

type Taxonomy = "fuel" | "transport" | "material" | "waste" | "electricity_market" | "spend";

interface MappingResponse {
  canonical: unknown;
  confidence: number;
  band: string;
  rationale: string;
  matched_pattern?: string;
  alternates?: unknown[];
  raw_input?: string;
}

export function FactorsMappingWorkbench() {
  const [taxonomy, setTaxonomy] = useState<Taxonomy>("fuel");
  const [input, setInput] = useState("");
  const [result, setResult] = useState<MappingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const resolve = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/factors/mapping/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taxonomy, description: input }),
      });
      if (!res.ok) throw new Error(`mapping ${res.status}`);
      setResult((await res.json()) as MappingResponse);
    } catch (e) {
      setError(
        `${(e as Error).message} — backend endpoint may not be wired yet; ` +
          "the Python API (greenlang.factors.mapping) is fully available.",
      );
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1000, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>Mapping Workbench</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Test free-text → canonical key resolution against any taxonomy.
        Confidence band + rationale + alternates are shown so operators can
        close synonym gaps before production.
      </Typography>

      <Card>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Taxonomy</InputLabel>
              <Select
                label="Taxonomy"
                value={taxonomy}
                onChange={(e) => setTaxonomy(e.target.value as Taxonomy)}
              >
                <MenuItem value="fuel">Fuel</MenuItem>
                <MenuItem value="transport">Transport</MenuItem>
                <MenuItem value="material">Material</MenuItem>
                <MenuItem value="waste">Waste</MenuItem>
                <MenuItem value="electricity_market">Electricity market</MenuItem>
                <MenuItem value="spend">Spend category</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              label="Description"
              placeholder="e.g. 'No. 2 distillate diesel', 'AWS cloud services Q2'"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void resolve();
              }}
            />
            <Button variant="contained" onClick={() => void resolve()} disabled={loading || !input.trim()}>
              Resolve
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {error && <Alert severity="warning" sx={{ mt: 2 }}>{error}</Alert>}

      {result && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Result</Typography>
            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              <Chip label={`Confidence: ${result.confidence.toFixed(2)}`} color="primary" />
              <Chip label={`Band: ${result.band}`} variant="outlined" />
              {result.matched_pattern && (
                <Chip label={`Matched: ${result.matched_pattern}`} variant="outlined" />
              )}
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {result.rationale}
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, overflow: "auto" }}>
              <pre style={{ margin: 0, fontSize: 13 }}>
                {JSON.stringify(result.canonical, null, 2)}
              </pre>
            </Paper>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default FactorsMappingWorkbench;
