/**
 * PipelineMonitor - Page for monitoring EUDR compliance pipelines.
 *
 * Shows active pipeline progress, "Start Pipeline" button, pipeline
 * execution history table, and per-stage metrics (average duration,
 * success rate).
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Grid,
  Card,
  CardContent,
  Paper,
  CircularProgress,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  Autocomplete,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  SelectChangeEvent,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SpeedIcon from '@mui/icons-material/Speed';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import TimerIcon from '@mui/icons-material/Timer';
import PipelineProgress from '../components/pipeline/PipelineProgress';
import PipelineHistory from '../components/pipeline/PipelineHistory';
import apiClient from '../services/api';
import type { PipelineRun, Supplier, EUDRCommodity, PipelineStage } from '../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMMODITIES: { value: EUDRCommodity; label: string }[] = [
  { value: 'cattle' as EUDRCommodity, label: 'Cattle' },
  { value: 'cocoa' as EUDRCommodity, label: 'Cocoa' },
  { value: 'coffee' as EUDRCommodity, label: 'Coffee' },
  { value: 'oil_palm' as EUDRCommodity, label: 'Oil Palm' },
  { value: 'rubber' as EUDRCommodity, label: 'Rubber' },
  { value: 'soya' as EUDRCommodity, label: 'Soya' },
  { value: 'wood' as EUDRCommodity, label: 'Wood' },
];

const STAGE_LABELS: Record<string, string> = {
  data_collection: 'Data Intake',
  geo_validation: 'Geo Validation',
  deforestation_check: 'Deforestation Risk',
  risk_assessment: 'Risk Assessment',
  dds_generation: 'DDS Reporting',
  final_review: 'Final Review',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PipelineMonitor: React.FC = () => {
  // Pipeline data
  const [activePipelines, setActivePipelines] = useState<PipelineRun[]>([]);
  const [history, setHistory] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Start pipeline dialog
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [selectedSupplier, setSelectedSupplier] = useState<Supplier | null>(null);
  const [selectedCommodity, setSelectedCommodity] = useState<string>('');
  const [startLoading, setStartLoading] = useState(false);

  // Snackbar
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false, message: '', severity: 'success',
  });

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const historyRes = await apiClient.getPipelineHistory({ per_page: 50 });
      const allRuns = historyRes.items;
      setActivePipelines(allRuns.filter((r) => r.status === 'running'));
      setHistory(allRuns);
    } catch {
      setError('Failed to load pipeline data.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Fetch suppliers for dialog
  useEffect(() => {
    apiClient
      .getSuppliers({ per_page: 500 })
      .then((res) => setSuppliers(res.items))
      .catch(() => {});
  }, []);

  // Polling for active pipelines
  useEffect(() => {
    if (activePipelines.length === 0) return;
    const interval = setInterval(() => {
      fetchData();
    }, 5000);
    return () => clearInterval(interval);
  }, [activePipelines.length, fetchData]);

  // Start pipeline
  const handleStartPipeline = async () => {
    if (!selectedSupplier || !selectedCommodity) return;
    try {
      setStartLoading(true);
      await apiClient.startPipeline({
        supplier_id: selectedSupplier.id,
        commodity: selectedCommodity as EUDRCommodity,
      });
      setSnackbar({ open: true, message: 'Pipeline started successfully.', severity: 'success' });
      setStartDialogOpen(false);
      setSelectedSupplier(null);
      setSelectedCommodity('');
      fetchData();
    } catch {
      setSnackbar({ open: true, message: 'Failed to start pipeline.', severity: 'error' });
    } finally {
      setStartLoading(false);
    }
  };

  // Retry / Cancel
  const handleRetry = async (run: PipelineRun) => {
    try {
      await apiClient.retryPipeline(run.id);
      setSnackbar({ open: true, message: 'Pipeline retrying...', severity: 'success' });
      fetchData();
    } catch {
      setSnackbar({ open: true, message: 'Retry failed.', severity: 'error' });
    }
  };

  const handleCancel = async (run: PipelineRun) => {
    try {
      await apiClient.cancelPipeline(run.id);
      setSnackbar({ open: true, message: 'Pipeline cancelled.', severity: 'success' });
      fetchData();
    } catch {
      setSnackbar({ open: true, message: 'Cancel failed.', severity: 'error' });
    }
  };

  // Compute stage metrics from history
  const stageMetrics = (() => {
    const stages = ['data_collection', 'geo_validation', 'deforestation_check', 'risk_assessment', 'dds_generation'];
    return stages.map((stage) => {
      const results = history.flatMap((run) =>
        run.stages.filter((s) => s.stage === stage && s.status !== 'pending')
      );
      const completed = results.filter((r) => r.status === 'completed');
      const durations = completed
        .map((r) => r.duration_seconds)
        .filter((d): d is number => d !== null);
      const avgDuration = durations.length > 0
        ? durations.reduce((a, b) => a + b, 0) / durations.length
        : 0;
      const successRate = results.length > 0
        ? (completed.length / results.length) * 100
        : 0;

      return {
        stage,
        label: STAGE_LABELS[stage] ?? stage,
        avgDuration,
        successRate,
        totalRuns: results.length,
      };
    });
  })();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', pt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h4" fontWeight={700}>
          Pipeline Monitor
        </Typography>
        <Button
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={() => setStartDialogOpen(true)}
        >
          Start Pipeline
        </Button>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Active Pipelines */}
      {activePipelines.length > 0 && (
        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            Active Pipelines ({activePipelines.length})
          </Typography>
          <Grid container spacing={2}>
            {activePipelines.map((run) => (
              <Grid item xs={12} md={6} key={run.id}>
                <PipelineProgress pipelineRun={run} />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {activePipelines.length === 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No active pipelines. Click "Start Pipeline" to run a compliance check.
        </Alert>
      )}

      {/* Stage Metrics */}
      <Typography variant="h6" gutterBottom>
        Stage Metrics
      </Typography>
      <Grid container spacing={2} mb={3}>
        {stageMetrics.map((metric) => (
          <Grid item xs={6} sm={4} md={2.4} key={metric.stage}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 1.5 }}>
                <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                  {metric.label}
                </Typography>
                <Stack direction="row" justifyContent="center" spacing={2}>
                  <Box>
                    <Stack direction="row" alignItems="center" spacing={0.5} justifyContent="center">
                      <TimerIcon sx={{ fontSize: 14 }} color="action" />
                      <Typography variant="body2" fontWeight={600}>
                        {metric.avgDuration > 0 ? `${metric.avgDuration.toFixed(0)}s` : '-'}
                      </Typography>
                    </Stack>
                    <Typography variant="caption" color="text.secondary">Avg Time</Typography>
                  </Box>
                  <Box>
                    <Stack direction="row" alignItems="center" spacing={0.5} justifyContent="center">
                      <CheckCircleIcon sx={{ fontSize: 14 }} color={metric.successRate >= 90 ? 'success' : 'warning'} />
                      <Typography variant="body2" fontWeight={600}>
                        {metric.successRate > 0 ? `${metric.successRate.toFixed(0)}%` : '-'}
                      </Typography>
                    </Stack>
                    <Typography variant="caption" color="text.secondary">Success</Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Pipeline History */}
      <PipelineHistory
        history={history}
        onRetry={handleRetry}
        onCancel={handleCancel}
      />

      {/* Start Pipeline Dialog */}
      <Dialog
        open={startDialogOpen}
        onClose={() => setStartDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Start Compliance Pipeline</DialogTitle>
        <DialogContent>
          <Stack spacing={2} mt={1}>
            <Autocomplete
              options={suppliers}
              getOptionLabel={(opt) => `${opt.name} (${opt.country})`}
              value={selectedSupplier}
              onChange={(_, val) => setSelectedSupplier(val)}
              renderInput={(params) => (
                <TextField {...params} label="Supplier" placeholder="Search supplier..." />
              )}
            />
            <FormControl fullWidth>
              <InputLabel>Commodity</InputLabel>
              <Select
                value={selectedCommodity}
                label="Commodity"
                onChange={(e: SelectChangeEvent) => setSelectedCommodity(e.target.value)}
              >
                {COMMODITIES.filter(
                  (c) => !selectedSupplier || selectedSupplier.commodities.includes(c.value)
                ).map((c) => (
                  <MenuItem key={c.value} value={c.value}>{c.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Divider />
            <Stack direction="row" justifyContent="flex-end" spacing={1}>
              <Button variant="outlined" onClick={() => setStartDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartPipeline}
                disabled={!selectedSupplier || !selectedCommodity || startLoading}
              >
                {startLoading ? 'Starting...' : 'Start Pipeline'}
              </Button>
            </Stack>
          </Stack>
        </DialogContent>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setSnackbar((s) => ({ ...s, open: false }))} severity={snackbar.severity} variant="filled">
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default PipelineMonitor;
