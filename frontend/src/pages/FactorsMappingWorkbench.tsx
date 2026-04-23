/**
 * F7.2 / Track B-5 — Mapping workbench.
 *
 * Wired to:
 *   POST /v1/admin/mapping/suggest    — get suggested family + canonical key
 *   POST /v1/admin/mapping/confirm    — promote a suggestion to a saved rule
 *
 * Lets operators paste a free-text activity / spend line and see what the
 * mapping layer (Phase F4) resolves; the Confirm action persists the
 * suggested mapping as a new rule (closes synonym gaps before they ship).
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
import {
  FactorsApiError,
  confirmMapping,
  suggestMapping,
  type MappingSuggestion,
} from "../lib/api/factorsClient";

type Taxonomy =
  | "fuel"
  | "transport"
  | "material"
  | "waste"
  | "electricity_market"
  | "spend";

const TAXONOMIES: Array<{ value: Taxonomy; label: string }> = [
  { value: "fuel", label: "Fuel" },
  { value: "transport", label: "Transport" },
  { value: "material", label: "Material" },
  { value: "waste", label: "Waste" },
  { value: "electricity_market", label: "Electricity market" },
  { value: "spend", label: "Spend category" },
];

export function FactorsMappingWorkbench() {
  const [taxonomy, setTaxonomy] = useState<Taxonomy>("fuel");
  const [input, setInput] = useState("");
  const [result, setResult] = useState<MappingSuggestion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [confirmedRuleId, setConfirmedRuleId] = useState<string | null>(null);

  const resolve = async () => {
    setLoading(true);
    setError(null);
    setConfirmedRuleId(null);
    try {
      const r = await suggestMapping(taxonomy, input);
      setResult(r);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const confirm = async () => {
    if (!result) return;
    setConfirming(true);
    setError(null);
    try {
      const { rule_id } = await confirmMapping({
        taxonomy,
        description: input,
        family: result.family,
        canonical_key: result.canonical_key,
      });
      setConfirmedRuleId(rule_id);
    } catch (e) {
      setError(e instanceof FactorsApiError ? e.userMessage : (e as Error).message);
    } finally {
      setConfirming(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1000, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Mapping Workbench
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Test free-text → canonical-key resolution against any taxonomy. Confirm to persist as a
        new mapping rule so the same input resolves consistently in production.
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
                {TAXONOMIES.map((t) => (
                  <MenuItem key={t.value} value={t.value}>
                    {t.label}
                  </MenuItem>
                ))}
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
            <Button
              variant="contained"
              onClick={() => void resolve()}
              disabled={loading || !input.trim()}
            >
              {loading ? "Resolving…" : "Resolve"}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }} role="alert">
          {error}
        </Alert>
      )}

      {confirmedRuleId && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Saved as rule <code>{confirmedRuleId}</code>. The same description will now resolve
          deterministically.
        </Alert>
      )}

      {result && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
              <Typography variant="h6">Result</Typography>
              <Chip
                label={`Confidence ${(result.confidence * 100).toFixed(0)}%`}
                color={
                  result.confidence >= 0.8
                    ? "success"
                    : result.confidence >= 0.5
                      ? "warning"
                      : "error"
                }
              />
              <Chip label={`Band: ${result.band}`} variant="outlined" />
              {result.matched_pattern && (
                <Chip label={`Matched: ${result.matched_pattern}`} variant="outlined" />
              )}
              <Box sx={{ flex: 1 }} />
              <Button
                variant="contained"
                color="primary"
                onClick={() => void confirm()}
                disabled={confirming}
              >
                {confirming ? "Saving…" : "Confirm + Save Rule"}
              </Button>
            </Stack>

            <Typography variant="subtitle2">Family</Typography>
            <Typography sx={{ mb: 2 }}><code>{result.family}</code></Typography>

            <Typography variant="subtitle2">Rationale</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {result.rationale ?? "—"}
            </Typography>

            <Typography variant="subtitle2">Canonical key</Typography>
            <Paper variant="outlined" sx={{ p: 2, overflow: "auto", mb: 2 }}>
              <pre style={{ margin: 0, fontSize: 13 }}>
                {JSON.stringify(result.canonical_key, null, 2)}
              </pre>
            </Paper>

            {result.alternates && result.alternates.length > 0 && (
              <>
                <Typography variant="subtitle2">Alternates</Typography>
                <Stack spacing={1} sx={{ mt: 1 }}>
                  {result.alternates.map((a, idx) => (
                    <Paper key={idx} variant="outlined" sx={{ p: 1 }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip size="small" label={`${(a.confidence * 100).toFixed(0)}%`} />
                        <code>{a.family}</code>
                      </Stack>
                      <pre style={{ margin: "4px 0 0 0", fontSize: 12 }}>
                        {JSON.stringify(a.canonical_key, null, 2)}
                      </pre>
                    </Paper>
                  ))}
                </Stack>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default FactorsMappingWorkbench;
