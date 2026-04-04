/** Shared pipeline DAG stages and evidence resolution for run visualizations. */
export type StageId = "validate" | "compute" | "policy" | "export" | "audit";

export interface PipelineStageDef {
  id: StageId;
  label: string;
  /** Prefer artifacts whose path matches any of these substrings (first match wins). */
  evidenceHints: string[];
}

export const PIPELINE_STAGES: PipelineStageDef[] = [
  { id: "validate", label: "Validate", evidenceHints: ["manifest", "input", "schema", "validation"] },
  { id: "compute", label: "Compute", evidenceHints: ["report", "inventory", "calc", "output"] },
  { id: "policy", label: "Policy", evidenceHints: ["policy", "bundle", "rego"] },
  { id: "export", label: "Export", evidenceHints: ["xml", "export", "xbrl"] },
  { id: "audit", label: "Audit", evidenceHints: ["audit", "checksum", "manifest"] }
];

export function pickEvidenceArtifact(artifacts: string[], hints: string[]): string | undefined {
  const lower = artifacts.map((a) => a.toLowerCase());
  for (const hint of hints) {
    const h = hint.toLowerCase();
    const idx = lower.findIndex((path) => path.includes(h));
    if (idx >= 0) return artifacts[idx];
  }
  return artifacts[0];
}

export function stageCompletion(
  runState: string | undefined,
  success: boolean | undefined,
  canExport: boolean | undefined,
  stageIndex: number
): number {
  if (runState === "failed") {
    return stageIndex === 0 ? 100 : Math.max(0, 100 - stageIndex * 25);
  }
  if (runState === "blocked") {
    const caps = [100, 100, 100, 55, 40];
    return caps[stageIndex] ?? 40;
  }
  if (runState === "partial_success") {
    const caps = [100, 100, 85, canExport === false ? 50 : 90, 75];
    return caps[stageIndex] ?? 70;
  }
  if (success === false) {
    return stageIndex === 0 ? 100 : 70 - stageIndex * 10;
  }
  if (canExport === false) {
    const caps = [100, 100, 100, 60, 80];
    return caps[stageIndex] ?? 80;
  }
  return 100;
}
