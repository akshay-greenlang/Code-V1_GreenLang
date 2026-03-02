/**
 * PipelineProgress - Visual 5-stage pipeline progress indicator.
 *
 * MUI Stepper showing: Intake -> Geo Validation -> Deforestation Risk ->
 * Doc Verification -> DDS Reporting. Color-coded by status with duration
 * per stage, error messages, and overall progress percentage.
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Stack,
  Chip,
  LinearProgress,
  Alert,
  CircularProgress,
  Paper,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import type { PipelineRun, StageResult, PipelineStage } from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STAGE_LABELS: Record<PipelineStage, string> = {
  data_collection: 'Data Intake',
  geo_validation: 'Geo Validation',
  deforestation_check: 'Deforestation Risk',
  risk_assessment: 'Risk Assessment',
  dds_generation: 'DDS Reporting',
  final_review: 'Final Review',
};

/** The 5 primary stages displayed in the stepper. */
const DISPLAY_STAGES: PipelineStage[] = [
  'data_collection' as PipelineStage,
  'geo_validation' as PipelineStage,
  'deforestation_check' as PipelineStage,
  'risk_assessment' as PipelineStage,
  'dds_generation' as PipelineStage,
];

const STATUS_COLORS = {
  completed: '#4caf50',
  running: '#1976d2',
  pending: '#bdbdbd',
  failed: '#f44336',
  skipped: '#9e9e9e',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PipelineProgressProps {
  pipelineRun: PipelineRun;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function getStageResult(
  stages: StageResult[],
  stageKey: PipelineStage
): StageResult | undefined {
  return stages.find((s) => s.stage === stageKey);
}

function StageIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircleIcon sx={{ color: STATUS_COLORS.completed }} />;
    case 'running':
      return <CircularProgress size={20} thickness={5} />;
    case 'failed':
      return <ErrorIcon sx={{ color: STATUS_COLORS.failed }} />;
    case 'skipped':
      return <SkipNextIcon sx={{ color: STATUS_COLORS.skipped }} />;
    default:
      return <HourglassEmptyIcon sx={{ color: STATUS_COLORS.pending }} />;
  }
}

function overallStatusChip(status: PipelineRun['status']) {
  const map: Record<string, { color: 'success' | 'info' | 'error' | 'default'; label: string }> = {
    pending: { color: 'default', label: 'Pending' },
    running: { color: 'info', label: 'Running' },
    completed: { color: 'success', label: 'Completed' },
    failed: { color: 'error', label: 'Failed' },
    cancelled: { color: 'default', label: 'Cancelled' },
  };
  const meta = map[status] ?? map.pending;
  return <Chip label={meta.label} color={meta.color} size="small" sx={{ fontWeight: 600 }} />;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PipelineProgress: React.FC<PipelineProgressProps> = ({ pipelineRun }) => {
  // Determine active step index
  const activeStepIndex = DISPLAY_STAGES.findIndex(
    (s) => s === pipelineRun.current_stage
  );

  // Completed step count
  const completedCount = DISPLAY_STAGES.filter((s) => {
    const result = getStageResult(pipelineRun.stages, s);
    return result?.status === 'completed';
  }).length;

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
          <Box>
            <Typography variant="h6">Pipeline Progress</Typography>
            <Typography variant="body2" color="text.secondary">
              {pipelineRun.supplier_name} -{' '}
              {pipelineRun.commodity.replace('_', ' ')}
            </Typography>
          </Box>
          <Stack alignItems="flex-end" spacing={0.5}>
            {overallStatusChip(pipelineRun.status)}
            <Typography variant="caption" color="text.secondary">
              Run ID: {pipelineRun.id.slice(0, 8)}
            </Typography>
          </Stack>
        </Stack>

        {/* Overall progress bar */}
        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
          <LinearProgress
            variant="determinate"
            value={pipelineRun.progress_percentage}
            sx={{ flex: 1, height: 10, borderRadius: 1 }}
            color={pipelineRun.status === 'failed' ? 'error' : 'primary'}
          />
          <Typography variant="body2" fontWeight={600} sx={{ minWidth: 40 }}>
            {pipelineRun.progress_percentage}%
          </Typography>
        </Stack>

        {/* Pipeline error */}
        {pipelineRun.error_message && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {pipelineRun.error_message}
          </Alert>
        )}

        {/* Stepper */}
        <Stepper
          activeStep={activeStepIndex >= 0 ? activeStepIndex : completedCount}
          orientation="vertical"
        >
          {DISPLAY_STAGES.map((stageKey) => {
            const result = getStageResult(pipelineRun.stages, stageKey);
            const status = result?.status ?? 'pending';

            return (
              <Step
                key={stageKey}
                completed={status === 'completed'}
                active={status === 'running'}
              >
                <StepLabel
                  icon={<StageIcon status={status} />}
                  optional={
                    <Typography variant="caption" color="text.secondary">
                      {formatDuration(result?.duration_seconds ?? null)}
                    </Typography>
                  }
                  error={status === 'failed'}
                >
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <Typography
                      variant="body2"
                      fontWeight={status === 'running' ? 700 : 500}
                      color={status === 'failed' ? 'error.main' : 'text.primary'}
                    >
                      {STAGE_LABELS[stageKey]}
                    </Typography>
                    {status === 'running' && (
                      <Chip label="In Progress" size="small" color="info" sx={{ height: 20, fontSize: 10 }} />
                    )}
                    {status === 'skipped' && (
                      <Chip label="Skipped" size="small" variant="outlined" sx={{ height: 20, fontSize: 10 }} />
                    )}
                  </Stack>
                </StepLabel>
                <StepContent>
                  {result?.output_summary && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                      {result.output_summary}
                    </Typography>
                  )}
                  {result?.errors && result.errors.length > 0 && (
                    <Alert severity="error" variant="outlined" sx={{ mt: 0.5, py: 0 }}>
                      {result.errors.map((err, i) => (
                        <Typography key={i} variant="caption" display="block">
                          {err}
                        </Typography>
                      ))}
                    </Alert>
                  )}
                  {result?.warnings && result.warnings.length > 0 && (
                    <Alert severity="warning" variant="outlined" sx={{ mt: 0.5, py: 0 }}>
                      {result.warnings.map((warn, i) => (
                        <Typography key={i} variant="caption" display="block">
                          {warn}
                        </Typography>
                      ))}
                    </Alert>
                  )}
                </StepContent>
              </Step>
            );
          })}
        </Stepper>

        {/* Timing footer */}
        <Stack direction="row" spacing={2} mt={2} justifyContent="flex-end">
          <Typography variant="caption" color="text.secondary">
            Started: {new Date(pipelineRun.started_at).toLocaleString('en-GB')}
          </Typography>
          {pipelineRun.completed_at && (
            <Typography variant="caption" color="text.secondary">
              Completed: {new Date(pipelineRun.completed_at).toLocaleString('en-GB')}
            </Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default PipelineProgress;
