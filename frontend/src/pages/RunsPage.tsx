import { useEffect, useMemo, useState } from "react";
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
import Typography from "@mui/material/Typography";
import { fetchArtifactText, listRuns, runArtifactUrl, runBundleUrl } from "../api";
import type { RunRecord } from "../types";

const stageOrder = ["validate", "compute", "policy", "export", "audit"] as const;

function stageProgress(run: RunRecord): number[] {
  const ok = run.success ?? run.status === "completed";
  if (!ok) return [100, 85, 60, 35, 20];
  return [100, 100, 100, run.can_export === false ? 70 : 100, 100];
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export function RunsPage() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [leftRun, setLeftRun] = useState("");
  const [rightRun, setRightRun] = useState("");
  const [selectedArtifact, setSelectedArtifact] = useState("");
  const [diffBusy, setDiffBusy] = useState(false);
  const [leftChecksum, setLeftChecksum] = useState<string | null>(null);
  const [rightChecksum, setRightChecksum] = useState<string | null>(null);
  const [diffPreview, setDiffPreview] = useState<string>("");

  const runsById = useMemo(() => new Map(runs.map((run) => [run.run_id, run])), [runs]);
  const leftArtifacts = useMemo(() => runsById.get(leftRun)?.artifacts ?? [], [runsById, leftRun]);
  const rightArtifacts = useMemo(() => runsById.get(rightRun)?.artifacts ?? [], [runsById, rightRun]);
  const commonArtifacts = useMemo(() => {
    const rightSet = new Set(rightArtifacts);
    return leftArtifacts.filter((artifact) => rightSet.has(artifact));
  }, [leftArtifacts, rightArtifacts]);

  useEffect(() => {
    listRuns().then(setRuns).catch((e) => setError(e instanceof Error ? e.message : "Failed to load runs"));
  }, []);

  useEffect(() => {
    setSelectedArtifact(commonArtifacts[0] ?? "");
    setLeftChecksum(null);
    setRightChecksum(null);
    setDiffPreview("");
  }, [leftRun, rightRun, commonArtifacts]);

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

  return (
    <Stack spacing={2}>
      <Typography variant="h5">Run Center</Typography>
      {error && <Alert severity="error">{error}</Alert>}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Run Graph Explorer</Typography>
          <Typography variant="body2">DAG stages with evidence links and export eligibility per run.</Typography>
        </CardContent>
      </Card>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Artifact Diff Viewer</Typography>
          <Stack direction="row" spacing={1} mt={1}>
            <Select size="small" displayEmpty value={leftRun} onChange={(e) => setLeftRun(e.target.value)}>
              <MenuItem value="">Run A</MenuItem>
              {runs.map((r) => (
                <MenuItem key={`a-${r.run_id}`} value={r.run_id}>{r.run_id}</MenuItem>
              ))}
            </Select>
            <Select size="small" displayEmpty value={rightRun} onChange={(e) => setRightRun(e.target.value)}>
              <MenuItem value="">Run B</MenuItem>
              {runs.map((r) => (
                <MenuItem key={`b-${r.run_id}`} value={r.run_id}>{r.run_id}</MenuItem>
              ))}
            </Select>
            <Select size="small" displayEmpty value={selectedArtifact} onChange={(e) => setSelectedArtifact(e.target.value)}>
              <MenuItem value="">Artifact</MenuItem>
              {commonArtifacts.map((artifact) => (
                <MenuItem key={artifact} value={artifact}>{artifact}</MenuItem>
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
              <Typography variant="caption" display="block">Run B checksum: {rightChecksum ?? "n/a"}</Typography>
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
      {runs.map((run) => (
        <Card key={run.run_id} variant="outlined">
          <CardContent>
            <Typography variant="body2">{run.app_id || "unknown"} - {run.run_id}</Typography>
            <Typography variant="caption">
              status: {run.status} | mode: {run.execution_mode ?? "unknown"} | export: {run.can_export === false ? "blocked" : "allowed"}
            </Typography>
            <Stack spacing={0.75} mt={1.5}>
              {stageOrder.map((stage, index) => (
                <Box key={`${run.run_id}-${stage}`}>
                  <Typography variant="caption">{stage}</Typography>
                  <LinearProgress variant="determinate" value={stageProgress(run)[index]} />
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
