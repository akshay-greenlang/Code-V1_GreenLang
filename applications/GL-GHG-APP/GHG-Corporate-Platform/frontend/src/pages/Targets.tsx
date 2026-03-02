/**
 * Targets Page - Emission reduction targets and SBTi alignment
 *
 * Composes a "Set Target" button/dialog, TargetProgress panel with
 * forecast trajectories, SBTiAlignment checker, and a gap analysis
 * summary section.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Alert,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Add, TrackChanges } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import {
  fetchTargets,
  fetchTargetProgress,
  checkSBTi,
  setTarget,
} from '../store/slices/targetsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import TargetProgressComponent from '../components/targets/TargetProgress';
import SBTiAlignment from '../components/targets/SBTiAlignment';
import { TargetType, SBTiPathway } from '../types';
import type { SetTargetRequest } from '../types';

const DEMO_ORG_ID = 'demo-org';

const TargetsPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { targets, progress, sbtiCheck, loading, error } = useAppSelector(
    (state) => state.targets
  );
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newTarget, setNewTarget] = useState<Partial<SetTargetRequest>>({
    name: '',
    target_type: TargetType.ABSOLUTE,
    scope_coverage: [],
    base_year: new Date().getFullYear() - 1,
    target_year: new Date().getFullYear() + 9,
    target_reduction_percent: 42,
    sbti_pathway: SBTiPathway.ONE_POINT_FIVE,
    intensity_metric: null,
    interim_targets: [],
  });

  useEffect(() => {
    dispatch(fetchTargets(DEMO_ORG_ID));
  }, [dispatch]);

  useEffect(() => {
    targets.forEach((t) => {
      if (!progress[t.id]) {
        dispatch(fetchTargetProgress({ orgId: DEMO_ORG_ID, targetId: t.id }));
      }
    });
  }, [targets, dispatch, progress]);

  useEffect(() => {
    if (targets.length > 0 && !sbtiCheck) {
      dispatch(checkSBTi({ orgId: DEMO_ORG_ID, targetId: targets[0].id }));
    }
  }, [targets, sbtiCheck, dispatch]);

  const handleCreateTarget = () => {
    dispatch(setTarget({ orgId: DEMO_ORG_ID, payload: newTarget as SetTargetRequest }));
    setDialogOpen(false);
  };

  if (loading && targets.length === 0) return <LoadingSpinner message="Loading targets..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Emission Reduction Targets</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setDialogOpen(true)}
        >
          Set Target
        </Button>
      </Box>

      {/* Target progress */}
      <Box sx={{ mb: 3 }}>
        <TargetProgressComponent targets={targets} progress={progress} />
      </Box>

      {/* SBTi alignment */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <SBTiAlignment
            target={targets.length > 0 ? targets[0] : null}
            sbtiCheck={sbtiCheck}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          {/* Gap analysis summary */}
          {targets.length > 0 && progress[targets[0].id] && (
            <Box>
              <Typography variant="h6" gutterBottom>Gap Analysis</Typography>
              <Alert
                severity={progress[targets[0].id].on_track ? 'success' : 'warning'}
                sx={{ mb: 2 }}
              >
                {progress[targets[0].id].on_track
                  ? 'Current trajectory is on track to meet the reduction target.'
                  : `Current trajectory shows a gap of ${Math.abs(progress[targets[0].id].gap_to_target).toFixed(0)} tCO2e to the reduction target.`}
              </Alert>
              <Typography variant="body2" color="text.secondary">
                Required annual reduction: {progress[targets[0].id].required_annual_reduction.toFixed(1)}%/yr
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Actual annual reduction: {progress[targets[0].id].actual_annual_reduction.toFixed(1)}%/yr
              </Typography>
            </Box>
          )}
        </Grid>
      </Grid>

      {/* Create target dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Set Reduction Target</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '16px !important' }}>
          <TextField
            label="Target Name"
            fullWidth
            value={newTarget.name || ''}
            onChange={(e) => setNewTarget({ ...newTarget, name: e.target.value })}
          />
          <FormControl fullWidth>
            <InputLabel>Target Type</InputLabel>
            <Select
              value={newTarget.target_type || TargetType.ABSOLUTE}
              label="Target Type"
              onChange={(e) => setNewTarget({ ...newTarget, target_type: e.target.value as TargetType })}
            >
              <MenuItem value={TargetType.ABSOLUTE}>Absolute</MenuItem>
              <MenuItem value={TargetType.INTENSITY}>Intensity</MenuItem>
            </Select>
          </FormControl>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="Base Year"
                type="number"
                fullWidth
                value={newTarget.base_year || ''}
                onChange={(e) => setNewTarget({ ...newTarget, base_year: parseInt(e.target.value) })}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="Target Year"
                type="number"
                fullWidth
                value={newTarget.target_year || ''}
                onChange={(e) => setNewTarget({ ...newTarget, target_year: parseInt(e.target.value) })}
              />
            </Grid>
          </Grid>
          <TextField
            label="Reduction %"
            type="number"
            fullWidth
            value={newTarget.target_reduction_percent || ''}
            onChange={(e) => setNewTarget({ ...newTarget, target_reduction_percent: parseFloat(e.target.value) })}
          />
          <FormControl fullWidth>
            <InputLabel>SBTi Pathway</InputLabel>
            <Select
              value={newTarget.sbti_pathway || ''}
              label="SBTi Pathway"
              onChange={(e) => setNewTarget({ ...newTarget, sbti_pathway: e.target.value as SBTiPathway })}
            >
              <MenuItem value={SBTiPathway.ONE_POINT_FIVE}>1.5C</MenuItem>
              <MenuItem value={SBTiPathway.WELL_BELOW_TWO}>Well Below 2C</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreateTarget} disabled={!newTarget.name}>
            Create Target
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TargetsPage;
