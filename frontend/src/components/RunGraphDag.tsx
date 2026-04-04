import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { shellColorTokens } from "@greenlang/shell-ui";
import { PIPELINE_STAGES, pickEvidenceArtifact, type StageId } from "../pipelineStages";

const NODE_R = 22;
const W = 520;
const H = 120;

interface Props {
  runId: string;
  artifacts: string[];
  runState?: string;
  selectedStage: StageId | null;
  onSelectStage: (stage: StageId) => void;
  artifactUrl: (path: string) => string;
}

export function RunGraphDag({
  runId,
  artifacts,
  runState,
  selectedStage,
  onSelectStage,
  artifactUrl
}: Props) {
  const n = PIPELINE_STAGES.length;
  const gap = (W - 40 - 2 * NODE_R) / Math.max(1, n - 1);
  const y = H / 2;

  return (
    <Box
      role="group"
      aria-label={`Run pipeline graph for ${runId}`}
      sx={{ width: "100%", overflowX: "auto" }}
    >
      <Typography variant="caption" color="text.secondary" component="p" id={`dag-help-${runId}`}>
        Use arrow keys while a stage is focused to move across the pipeline. Enter opens the evidence
        artifact when available. Run state: {runState ?? "unknown"}.
      </Typography>
      <svg
        width={W}
        height={H + 36}
        viewBox={`0 0 ${W} ${H + 36}`}
        role="img"
        aria-labelledby={`dag-title-${runId}`}
      >
        <title id={`dag-title-${runId}`}>Pipeline DAG for run {runId}</title>
        {PIPELINE_STAGES.map((stage, i) => {
          if (i === 0) return null;
          const x1 = 40 + (i - 1) * gap;
          const x2 = 40 + i * gap;
          return (
            <line
              key={`edge-${stage.id}`}
              x1={x1 + NODE_R}
              y1={y}
              x2={x2 - NODE_R}
              y2={y}
              stroke="rgba(255,255,255,0.25)"
              strokeWidth={2}
            />
          );
        })}
        {PIPELINE_STAGES.map((stage, i) => {
          const cx = 40 + i * gap;
          const focused = selectedStage === stage.id;
          const evidence = pickEvidenceArtifact(artifacts, stage.evidenceHints);
          return (
            <g key={stage.id}>
              <circle
                role="button"
                tabIndex={0}
                aria-label={`${stage.label} stage${evidence ? `, evidence ${evidence}` : ""}`}
                aria-pressed={focused}
                cx={cx}
                cy={y}
                r={NODE_R}
                fill={focused ? shellColorTokens.primary : "rgba(79,124,255,0.25)"}
                stroke={shellColorTokens.secondary}
                strokeWidth={focused ? 3 : 1}
                style={{ cursor: "pointer" }}
                onClick={() => onSelectStage(stage.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (evidence) {
                      window.open(artifactUrl(evidence), "_blank", "noopener,noreferrer");
                    } else {
                      onSelectStage(stage.id);
                    }
                  }
                  if (e.key === "ArrowRight" && i < n - 1) {
                    e.preventDefault();
                    onSelectStage(PIPELINE_STAGES[i + 1].id);
                  }
                  if (e.key === "ArrowLeft" && i > 0) {
                    e.preventDefault();
                    onSelectStage(PIPELINE_STAGES[i - 1].id);
                  }
                }}
              />
              <text
                x={cx}
                y={y + 4}
                textAnchor="middle"
                fill="#e8efff"
                fontSize={11}
                fontFamily="inherit"
                pointerEvents="none"
              >
                {stage.label}
              </text>
            </g>
          );
        })}
      </svg>
      {selectedStage && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          Selected: {PIPELINE_STAGES.find((s) => s.id === selectedStage)?.label}
          {(() => {
            const st = PIPELINE_STAGES.find((s) => s.id === selectedStage);
            const art = st ? pickEvidenceArtifact(artifacts, st.evidenceHints) : undefined;
            return art ? (
              <>
                {" · "}
                <Box component="a" href={artifactUrl(art)} sx={{ color: "secondary.main" }}>
                  Open evidence: {art}
                </Box>
              </>
            ) : (
              " · No matching artifact for this stage."
            );
          })()}
        </Typography>
      )}
    </Box>
  );
}
