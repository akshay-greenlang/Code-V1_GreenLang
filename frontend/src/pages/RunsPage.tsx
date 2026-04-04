import { useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Divider from "@mui/material/Divider";
import LinearProgress from "@mui/material/LinearProgress";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { RunStatusChip, apiStatusChipFromResponse } from "@greenlang/shell-ui";
import { fetchArtifactText, listRuns, runArtifactUrl, runBundleUrl } from "../api";
import { RunGraphDag } from "../components/RunGraphDag";
import { PIPELINE_STAGES, stageCompletion, type StageId } from "../pipelineStages";
import type { AppKey, RunRecord } from "../types";

const APP_KEYS = new Set<AppKey>(["cbam", "csrd", "vcci", "eudr", "ghg", "iso14064", "sb253", "taxonomy"]);

function dayStartUtcTs(isoDate: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(isoDate)) return null;
  const d = new Date(`${isoDate}T00:00:00.000Z`);
  return Number.isNaN(d.getTime()) ? null : d.getTime() / 1000;
}

function dayEndUtcTs(isoDate: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(isoDate)) return null;
  const d = new Date(`${isoDate}T23:59:59.999Z`);
  return Number.isNaN(d.getTime()) ? null : d.getTime() / 1000;
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export function RunsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const appFilterRaw = searchParams.get("app");
  const appFilter =
    appFilterRaw && APP_KEYS.has(appFilterRaw as AppKey) ? (appFilterRaw as AppKey) : null;
  const statusFilterRaw = searchParams.get("status")?.trim() ?? "";
  const qFilterRaw = searchParams.get("q")?.trim() ?? "";
  const sinceRaw = searchParams.get("since")?.trim() ?? "";
  const untilRaw = searchParams.get("until")?.trim() ?? "";

  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [leftRun, setLeftRun] = useState("");
  const [rightRun, setRightRun] = useState("");
  const [selectedArtifact, setSelectedArtifact] = useState("");
  const [diffBusy, setDiffBusy] = useState(false);
  const [leftChecksum, setLeftChecksum] = useState<string | null>(null);
  const [rightChecksum, setRightChecksum] = useState<string | null>(null);
  const [diffPreview, setDiffPreview] = useState<string>("");
  const [graphRunId, setGraphRunId] = useState<string>("");
  const [dagStage, setDagStage] = useState<StageId | null>("validate");

  const listQuery = useMemo(
    () => ({
      app_id: appFilter ?? undefined,
      status: statusFilterRaw || undefined,
      q: qFilterRaw || undefined,
      since_ts: sinceRaw ? dayStartUtcTs(sinceRaw) ?? undefined : undefined,
      until_ts: untilRaw ? dayEndUtcTs(untilRaw) ?? undefined : undefined
    }),
    [appFilter, statusFilterRaw, qFilterRaw, sinceRaw, untilRaw]
  );

  const filteredRuns = runs;

  const patchSearchParams = (patch: Record<string, string>) => {
    const next = new URLSearchParams(searchParams);
    Object.entries(patch).forEach(([k, v]) => {
      if (!v) next.delete(k);
      else next.set(k, v);
    });
    setSearchParams(next);
  };

  const runsById = useMemo(() => new Map(filteredRuns.map((run) => [run.run_id, run])), [filteredRuns]);
  const leftArtifacts = useMemo(() => runsById.get(leftRun)?.artifacts ?? [], [runsById, leftRun]);
  const rightArtifacts = useMemo(() => runsById.get(rightRun)?.artifacts ?? [], [runsById, rightRun]);
  const commonArtifacts = useMemo(() => {
    const rightSet = new Set(rightArtifacts);
    return leftArtifacts.filter((artifact) => rightSet.has(artifact));
  }, [leftArtifacts, rightArtifacts]);

  const graphRun = graphRunId ? runsById.get(graphRunId) : undefined;

  useEffect(() => {
    setError(null);
    listRuns(listQuery)
      .then(setRuns)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load runs"));
  }, [listQuery]);

  useEffect(() => {
    setSelectedArtifact(commonArtifacts[0] ?? "");
    setLeftChecksum(null);
    setRightChecksum(null);
    setDiffPreview("");
  }, [leftRun, rightRun, commonArtifacts]);

  useEffect(() => {
    if (!filteredRuns.length) {
      setGraphRunId("");
      return;
    }
    if (!graphRunId || !filteredRuns.some((r) => r.run_id === graphRunId)) {
      setGraphRunId(filteredRuns[0].run_id);
    }
  }, [filteredRuns, graphRunId]);

  useEffect(() => {
    if (leftRun && !filteredRuns.some((r) => r.run_id === leftRun)) setLeftRun("");
    if (rightRun && !filteredRuns.some((r) => r.run_id === rightRun)) setRightRun("");
  }, [filteredRuns, leftRun, rightRun]);

  const compareArtifact = async () => {
    if (!leftRun || !rightRun || !selectedArtifact) return;
    setDiffBusy(true);
    setError(null);
    try {
      const [leftText, rightText] = await Promise.all([
        fetchArtifactText(leftRun, selectedArtifact),
        fetchArtifactText(rightRun, selectedArtifact)
      ]);
      const [leftHash, rightHash] = await Promise.all([sha256Hex(leftText), sha256Hex(rightText)]);
      setLeftChecksum(leftHash);
      setRightChecksum(rightHash);
      if (leftText === rightText) {
        setDiffPreview("No differences detected in artifact content.");
      } else {
        const leftLines = leftText.split(/\r?\n/);
        const rightLines = rightText.split(/\r?\n/);
        const max = Math.max(leftLines.length, rightLines.length);
        const changes: string[] = [];
        for (let index = 0; index < max && changes.length < 10; index += 1) {
          if ((leftLines[index] ?? "") !== (rightLines[index] ?? "")) {
            changes.push(`L${index + 1}\n- ${leftLines[index] ?? "<missing>"}\n+ ${rightLines[index] ?? "<missing>"}`);
          }
        }
        setDiffPreview(changes.join("\n\n"));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Artifact diff failed");
    } finally {
      setDiffBusy(false);
    }
  };

  const hasActiveFilters =
    Boolean(appFilter) ||
    Boolean(statusFilterRaw) ||
    Boolean(qFilterRaw) ||
    Boolean(sinceRaw) ||
    Boolean(untilRaw);

  return (
    <Stack spacing={2}>
      <Typography variant="h5">Run Center</Typography>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Filters
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
            <Select
              size="small"
              displayEmpty
              value={appFilter ?? ""}
              onChange={(e) => patchSearchParams({ app: e.target.value as string })}
              inputProps={{ "aria-label": "Filter by application" }}
              sx={{ minWidth: 140 }}
            >
              <MenuItem value="">All apps</MenuItem>
              {Array.from(APP_KEYS).map((k) => (
                <MenuItem key={k} value={k}>
                  {k}
                </MenuItem>
              ))}
            </Select>
            <Select
              size="small"
              displayEmpty
              value={statusFilterRaw}
              onChange={(e) => patchSearchParams({ status: e.target.value as string })}
              inputProps={{ "aria-label": "Filter by run state or chip" }}
              sx={{ minWidth: 180 }}
            >
              <MenuItem value="">Any status</MenuItem>
              <MenuItem value="completed">completed</MenuItem>
              <MenuItem value="failed">failed</MenuItem>
              <MenuItem value="blocked">blocked</MenuItem>
              <MenuItem value="partial_success">partial_success</MenuItem>
              <MenuItem value="PASS">PASS chip</MenuItem>
              <MenuItem value="FAIL">FAIL chip</MenuItem>
              <MenuItem value="WARN">WARN chip</MenuItem>
            </Select>
            <TextField
              size="small"
              label="Search run / app"
              value={qFilterRaw}
              onChange={(e) => patchSearchParams({ q: e.target.value })}
              inputProps={{ "aria-label": "Search by run id or app id substring" }}
            />
            <TextField
              size="small"
              label="Since"
              type="date"
              InputLabelProps={{ shrink: true }}
              value={sinceRaw}
              onChange={(e) => patchSearchParams({ since: e.target.value })}
              inputProps={{ "aria-label": "Created on or after" }}
            />
            <TextField
              size="small"
              label="Until"
              type="date"
              InputLabelProps={{ shrink: true }}
              value={untilRaw}
              onChange={(e) => patchSearchParams({ until: e.target.value })}
              inputProps={{ "aria-label": "Created on or before" }}
            />
            {hasActiveFilters && (
              <Button component={RouterLink} to="/runs" size="small" variant="outlined">
                Clear filters
              </Button>
            )}
          </Stack>
        </CardContent>
      </Card>
      {hasActiveFilters && (
        <Alert severity="info" sx={{ py: 0.5 }}>
          Results reflect active filters (server-side query).{" "}
          <Button component={RouterLink} to="/runs" size="small">
            Clear all
          </Button>
        </Alert>
      )}
      {error && <Alert severity="error">{error}</Alert>}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Run graph explorer</Typography>
          <Typography variant="body2">
            Interactive pipeline DAG with evidence deep-links per stage (selected run below).
          </Typography>
          <Stack direction="row" spacing={1} mt={2} alignItems="center" flexWrap="wrap">
            <Typography variant="body2">Focus run:</Typography>
            <Select
              size="small"
              displayEmpty
              value={graphRunId}
              onChange={(e) => setGraphRunId(e.target.value)}
              inputProps={{ "aria-label": "Run for DAG explorer" }}
            >
              <MenuItem value="">Select run</MenuItem>
              {filteredRuns.map((r) => (
                <MenuItem key={`g-${r.run_id}`} value={r.run_id}>
                  {r.app_id || "app"} — {r.run_id.slice(0, 8)}…
                </MenuItem>
              ))}
            </Select>
          </Stack>
          {graphRun && (
            <Box sx={{ mt: 2 }}>
              <RunGraphDag
                runId={graphRun.run_id}
                artifacts={graphRun.artifacts ?? []}
                runState={graphRun.run_state}
                selectedStage={dagStage}
                onSelectStage={setDagStage}
                artifactUrl={(path) => runArtifactUrl(graphRun.run_id, path)}
              />
            </Box>
          )}
          {!graphRun && filteredRuns.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No runs yet — execute a workspace demo to populate the graph.
            </Typography>
          )}
        </CardContent>
      </Card>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Artifact diff viewer</Typography>
          <Stack direction="row" spacing={1} mt={1}>
            <Select
              size="small"
              displayEmpty
              value={leftRun}
              onChange={(e) => setLeftRun(e.target.value)}
              inputProps={{ "aria-label": "Artifact diff run A" }}
            >
              <MenuItem value="">Run A</MenuItem>
              {filteredRuns.map((r) => (
                <MenuItem key={`a-${r.run_id}`} value={r.run_id}>
                  {r.run_id}
                </MenuItem>
              ))}
            </Select>
            <Select
              size="small"
              displayEmpty
              value={rightRun}
              onChange={(e) => setRightRun(e.target.value)}
              inputProps={{ "aria-label": "Artifact diff run B" }}
            >
              <MenuItem value="">Run B</MenuItem>
              {filteredRuns.map((r) => (
                <MenuItem key={`b-${r.run_id}`} value={r.run_id}>
                  {r.run_id}
                </MenuItem>
              ))}
            </Select>
            <Select
              size="small"
              displayEmpty
              value={selectedArtifact}
              onChange={(e) => setSelectedArtifact(e.target.value)}
              inputProps={{ "aria-label": "Artifact diff common artifact" }}
            >
              <MenuItem value="">Artifact</MenuItem>
              {commonArtifacts.map((artifact) => (
                <MenuItem key={artifact} value={artifact}>
                  {artifact}
                </MenuItem>
              ))}
            </Select>
            <Button variant="contained" onClick={compareArtifact} disabled={!selectedArtifact || diffBusy}>
              Compare
            </Button>
          </Stack>
          <Typography variant="body2" mt={1}>
            {leftRun && rightRun
              ? `Comparing ${leftRun} vs ${rightRun} using checksum parity.`
              : "Select two runs for side-by-side artifact comparison."}
          </Typography>
          {diffBusy && <LinearProgress sx={{ mt: 1 }} />}
          {(leftChecksum || rightChecksum) && (
            <Box mt={1}>
              <Typography variant="caption">Run A checksum: {leftChecksum ?? "n/a"}</Typography>
              <Typography variant="caption" display="block">
                Run B checksum: {rightChecksum ?? "n/a"}
              </Typography>
              <Typography variant="caption" color={leftChecksum === rightChecksum ? "success.main" : "warning.main"}>
                {leftChecksum === rightChecksum ? "Checksum parity: match" : "Checksum parity: mismatch"}
              </Typography>
            </Box>
          )}
          {diffPreview && (
            <Box mt={1} sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 12 }}>
              {diffPreview}
            </Box>
          )}
        </CardContent>
      </Card>
      <Divider />
      {filteredRuns.map((run) => (
        <Card key={run.run_id} variant="outlined">
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap" useFlexGap>
              <Typography variant="body2">
                {run.app_id || "unknown"} - {run.run_id}
              </Typography>
              <RunStatusChip chip={apiStatusChipFromResponse(run.status_chip)} runState={run.run_state} />
            </Stack>
            <Typography variant="caption">
              status: {run.status} | mode: {run.execution_mode ?? "unknown"} | export:{" "}
              {run.can_export === false ? "blocked" : "allowed"} | lifecycle: {run.run_state ?? "n/a"}
            </Typography>
            <Stack spacing={0.75} mt={1.5}>
              {PIPELINE_STAGES.map((stage, index) => (
                <Box key={`${run.run_id}-${stage.id}`}>
                  <Typography variant="caption">{stage.id}</Typography>
                  <LinearProgress
                    variant="determinate"
                    value={stageCompletion(run.run_state, run.success, run.can_export, index)}
                  />
                </Box>
              ))}
            </Stack>
            <Stack direction="row" spacing={1} mt={1.5} flexWrap="wrap">
              <Button variant="outlined" size="small" href={runBundleUrl(run.run_id)}>
                Download Bundle
              </Button>
              {(run.artifacts ?? []).slice(0, 3).map((artifact) => (
                <Button key={`${run.run_id}-${artifact}`} variant="text" size="small" href={runArtifactUrl(run.run_id, artifact)}>
                  {artifact}
                </Button>
              ))}
            </Stack>
          </CardContent>
        </Card>
      ))}
    </Stack>
  );
}
