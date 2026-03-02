/**
 * PipelineHistory - Table of pipeline execution history.
 *
 * Displays pipeline runs with status badges, expandable per-stage details,
 * duration, and retry/cancel action buttons.
 */

import React, { useState, Fragment } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  Collapse,
  Stack,
  LinearProgress,
  Button,
} from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import ReplayIcon from '@mui/icons-material/Replay';
import CancelIcon from '@mui/icons-material/Cancel';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import type { PipelineRun, StageResult, PipelineStage } from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_CHIP_MAP: Record<string, { color: 'info' | 'success' | 'error' | 'default'; label: string }> = {
  running: { color: 'info', label: 'Running' },
  completed: { color: 'success', label: 'Completed' },
  failed: { color: 'error', label: 'Failed' },
  cancelled: { color: 'default', label: 'Cancelled' },
  pending: { color: 'default', label: 'Pending' },
};

const STAGE_LABELS: Record<string, string> = {
  data_collection: 'Data Intake',
  geo_validation: 'Geo Validation',
  deforestation_check: 'Deforestation Risk',
  risk_assessment: 'Risk Assessment',
  dds_generation: 'DDS Reporting',
  final_review: 'Final Review',
};

const STAGE_STATUS_ICONS: Record<string, React.ReactNode> = {
  completed: <CheckCircleIcon fontSize="small" color="success" />,
  running: <PlayCircleIcon fontSize="small" color="info" />,
  failed: <ErrorIcon fontSize="small" color="error" />,
  pending: <HourglassEmptyIcon fontSize="small" color="disabled" />,
  skipped: <HourglassEmptyIcon fontSize="small" color="disabled" />,
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PipelineHistoryProps {
  history: PipelineRun[];
  onRetry?: (run: PipelineRun) => void;
  onCancel?: (run: PipelineRun) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function computeDuration(run: PipelineRun): string {
  if (!run.started_at) return '-';
  const start = new Date(run.started_at).getTime();
  const end = run.completed_at
    ? new Date(run.completed_at).getTime()
    : Date.now();
  const diffSec = (end - start) / 1000;
  return formatDuration(diffSec);
}

function formatDateTime(d: string): string {
  return new Date(d).toLocaleString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ---------------------------------------------------------------------------
// Expandable Row
// ---------------------------------------------------------------------------

function ExpandableRow({
  run,
  onRetry,
  onCancel,
}: {
  run: PipelineRun;
  onRetry?: (run: PipelineRun) => void;
  onCancel?: (run: PipelineRun) => void;
}) {
  const [open, setOpen] = useState(false);
  const statusMeta = STATUS_CHIP_MAP[run.status] ?? STATUS_CHIP_MAP.pending;

  return (
    <Fragment>
      <TableRow hover sx={{ '& > *': { borderBottom: open ? 'none' : undefined } }}>
        <TableCell padding="checkbox">
          <IconButton size="small" onClick={() => setOpen(!open)}>
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell>
          <Typography variant="body2" fontWeight={500}>
            {run.id.slice(0, 12)}...
          </Typography>
        </TableCell>
        <TableCell>{run.supplier_name}</TableCell>
        <TableCell>{formatDateTime(run.started_at)}</TableCell>
        <TableCell>{computeDuration(run)}</TableCell>
        <TableCell>
          <Chip
            label={statusMeta.label}
            color={statusMeta.color}
            size="small"
            sx={{ fontWeight: 600 }}
          />
        </TableCell>
        <TableCell sx={{ textTransform: 'capitalize' }}>
          {STAGE_LABELS[run.current_stage] ?? run.current_stage.replace('_', ' ')}
        </TableCell>
        <TableCell align="center">
          <Stack direction="row" spacing={0.5} justifyContent="center">
            {run.status === 'failed' && onRetry && (
              <Tooltip title="Retry">
                <IconButton size="small" color="primary" onClick={() => onRetry(run)}>
                  <ReplayIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            {run.status === 'running' && onCancel && (
              <Tooltip title="Cancel">
                <IconButton size="small" color="error" onClick={() => onCancel(run)}>
                  <CancelIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Stack>
        </TableCell>
      </TableRow>

      {/* Expanded detail */}
      <TableRow>
        <TableCell colSpan={8} sx={{ py: 0 }}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Stage Details
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Stage</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Summary</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {run.stages.map((stage) => (
                    <TableRow key={stage.stage}>
                      <TableCell>
                        <Stack direction="row" alignItems="center" spacing={1}>
                          {STAGE_STATUS_ICONS[stage.status] ?? STAGE_STATUS_ICONS.pending}
                          <Typography variant="body2">
                            {STAGE_LABELS[stage.stage] ?? stage.stage.replace('_', ' ')}
                          </Typography>
                        </Stack>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{ textTransform: 'capitalize' }}
                          color={
                            stage.status === 'failed'
                              ? 'error.main'
                              : stage.status === 'completed'
                              ? 'success.main'
                              : 'text.secondary'
                          }
                        >
                          {stage.status}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {formatDuration(stage.duration_seconds)}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 300 }}>
                          {stage.output_summary || '-'}
                        </Typography>
                        {stage.errors.length > 0 && (
                          <Typography variant="caption" color="error.main">
                            {stage.errors[0]}
                          </Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Progress bar */}
              <Stack direction="row" alignItems="center" spacing={1} mt={1.5}>
                <LinearProgress
                  variant="determinate"
                  value={run.progress_percentage}
                  sx={{ flex: 1, height: 6, borderRadius: 1 }}
                  color={run.status === 'failed' ? 'error' : 'primary'}
                />
                <Typography variant="caption" fontWeight={600}>
                  {run.progress_percentage}%
                </Typography>
              </Stack>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </Fragment>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PipelineHistory: React.FC<PipelineHistoryProps> = ({
  history,
  onRetry,
  onCancel,
}) => {
  return (
    <Paper sx={{ width: '100%' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Pipeline Execution History</Typography>
      </Box>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox" />
              <TableCell>Run ID</TableCell>
              <TableCell>Supplier</TableCell>
              <TableCell>Started</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Current Stage</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {history.map((run) => (
              <ExpandableRow
                key={run.id}
                run={run}
                onRetry={onRetry}
                onCancel={onCancel}
              />
            ))}
            {history.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">
                    No pipeline executions found.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default PipelineHistory;
