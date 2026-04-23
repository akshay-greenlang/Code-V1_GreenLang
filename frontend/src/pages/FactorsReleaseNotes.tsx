/**
 * O9 / W4-D — Versioned release-notes feed.
 *
 * Wired to GET /v1/release-notes. Each entry includes a link back to
 * the diff viewer pre-filled with the previous edition so reviewers can
 * see exactly what changed.
 */
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import LinearProgress from "@mui/material/LinearProgress";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { Link as RouterLink } from "react-router-dom";
import {
  FactorsApiError,
  listReleaseNotes,
  type ReleaseNoteEntry,
} from "../lib/api/factorsClient";

export function FactorsReleaseNotes() {
  const [notes, setNotes] = useState<ReleaseNoteEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const r = await listReleaseNotes();
        if (!cancelled) {
          setNotes(r.notes ?? []);
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
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: 900, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Release Notes
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Per-edition changelog with cross-edition diff links. Use this to
        trace factor-value drift between certified editions.
      </Typography>

      {loading && <LinearProgress />}
      {error && (
        <Alert severity="error" role="alert">
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        {notes.map((n) => (
          <Card key={n.edition_id} data-testid={`release-note-${n.edition_id}`}>
            <CardContent>
              <Stack direction="row" spacing={1} alignItems="baseline" sx={{ mb: 1 }}>
                <Typography variant="h6">
                  <code>{n.edition_id}</code>
                </Typography>
                <Chip size="small" label={new Date(n.published_at).toLocaleDateString()} />
                {n.previous_edition_id && (
                  <Link
                    component={RouterLink}
                    to={`/factors/diff?from=${encodeURIComponent(n.previous_edition_id)}&to=${encodeURIComponent(n.edition_id)}`}
                    aria-label={`Diff from ${n.previous_edition_id} to ${n.edition_id}`}
                  >
                    diff vs {n.previous_edition_id}
                  </Link>
                )}
              </Stack>
              <Typography variant="body2" sx={{ mb: 1 }}>
                {n.summary}
              </Typography>
              {n.sections.map((s) => (
                <Box key={s.title} sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">{s.title}</Typography>
                  <Typography
                    component="pre"
                    variant="body2"
                    sx={{
                      whiteSpace: "pre-wrap",
                      fontFamily: "inherit",
                      m: 0,
                    }}
                  >
                    {s.body}
                  </Typography>
                </Box>
              ))}
              {n.factor_diff_url && (
                <Link href={n.factor_diff_url} target="_blank" rel="noreferrer">
                  Raw factor diff (JSON)
                </Link>
              )}
            </CardContent>
          </Card>
        ))}
        {!loading && notes.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No release notes published yet.
          </Typography>
        )}
      </Stack>
    </Box>
  );
}

export default FactorsReleaseNotes;
