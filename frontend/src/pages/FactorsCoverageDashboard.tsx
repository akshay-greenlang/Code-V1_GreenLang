/**
 * O8 / W4-D — Public coverage dashboard.
 *
 * Mounted at `/factors/coverage` WITHOUT auth. Renders Certified /
 * Preview / Connector-only counts per family × jurisdiction so prospects
 * can self-assess whether GreenLang covers their footprint.
 *
 * Mobile-first: the matrix degrades to stacked cards on small screens.
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import LinearProgress from "@mui/material/LinearProgress";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import useMediaQuery from "@mui/material/useMediaQuery";
import {
  FactorsApiError,
  getCoverage,
  type CoverageByFamily,
  type CoverageResponse,
  type CoverageTotals,
} from "../lib/api/factorsClient";

interface MatrixCell extends CoverageTotals {
  family: string;
  jurisdiction: string;
}

function deriveMatrix(resp: CoverageResponse): MatrixCell[] {
  const extra = resp as unknown as { by_family_jurisdiction?: MatrixCell[] };
  if (Array.isArray(extra.by_family_jurisdiction)) return extra.by_family_jurisdiction;
  return (resp.by_family ?? []).map((f: CoverageByFamily) => ({
    ...f,
    jurisdiction: "ALL",
  }));
}

export function FactorsCoverageDashboard() {
  const [data, setData] = useState<CoverageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const isMobile = useMediaQuery("(max-width:600px)");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const r = await getCoverage();
        if (!cancelled) {
          setData(r);
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
    const t = window.setInterval(load, 60_000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, []);

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 1200, mx: "auto" }} data-testid="coverage-dashboard">
      <Typography variant="h4" gutterBottom>
        Factors Catalog — Public Coverage
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Per-family × per-jurisdiction counts of <strong>Certified</strong>,{" "}
        <strong>Preview</strong>, and <strong>Connector-only</strong>{" "}
        factors in the current edition. Public — no authentication required.
        See <Link href="/pricing">pricing</Link> for the data-class matrix.
      </Typography>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" role="alert">
          {error}
        </Alert>
      )}

      {data && (
        <>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip
                  color="success"
                  label={`Certified: ${data.totals.certified.toLocaleString()}`}
                  data-testid="coverage-totals-certified"
                />
                <Chip
                  color="warning"
                  label={`Preview: ${data.totals.preview.toLocaleString()}`}
                  data-testid="coverage-totals-preview"
                />
                <Chip
                  color="default"
                  label={`Connector-only: ${data.totals.connector_only.toLocaleString()}`}
                  data-testid="coverage-totals-connector-only"
                />
                <Chip label={`Total: ${data.totals.all.toLocaleString()}`} />
              </Stack>
              <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
                Edition: <code>{data.edition_id}</code> · generated{" "}
                {new Date(data.generated_at).toLocaleString()}
              </Typography>
            </CardContent>
          </Card>

          {isMobile ? (
            <Stack spacing={1.5}>
              {deriveMatrix(data).map((c, idx) => (
                <Card key={`${c.family}-${c.jurisdiction}-${idx}`} variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">
                      <code>{c.family}</code> · <code>{c.jurisdiction}</code>
                    </Typography>
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
                      <Chip size="small" color="success" label={`C ${c.certified}`} />
                      <Chip size="small" color="warning" label={`P ${c.preview}`} />
                      <Chip size="small" label={`CO ${c.connector_only}`} />
                      <Chip size="small" variant="outlined" label={`Σ ${c.all}`} />
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Stack>
          ) : (
            <Card>
              <CardContent>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small" aria-label="Public coverage matrix">
                    <TableHead>
                      <TableRow>
                        <TableCell>Family</TableCell>
                        <TableCell>Jurisdiction</TableCell>
                        <TableCell align="right">Certified</TableCell>
                        <TableCell align="right">Preview</TableCell>
                        <TableCell align="right">Connector-only</TableCell>
                        <TableCell align="right">Total</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {deriveMatrix(data).map((c, idx) => (
                        <TableRow key={`${c.family}-${c.jurisdiction}-${idx}`} hover>
                          <TableCell><code>{c.family}</code></TableCell>
                          <TableCell><code>{c.jurisdiction}</code></TableCell>
                          <TableCell align="right">{c.certified.toLocaleString()}</TableCell>
                          <TableCell align="right">{c.preview.toLocaleString()}</TableCell>
                          <TableCell align="right">
                            {c.connector_only.toLocaleString()}
                          </TableCell>
                          <TableCell align="right"><strong>{c.all.toLocaleString()}</strong></TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
}

export default FactorsCoverageDashboard;
