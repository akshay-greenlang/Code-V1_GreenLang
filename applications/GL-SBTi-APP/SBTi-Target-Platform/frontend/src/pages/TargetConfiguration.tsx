/**
 * TargetConfiguration - Create, view, and manage SBTi targets.
 *
 * Shows target table, target form dialog, scope selector, method picker, and status timeline.
 */

import React, { useEffect, useState } from 'react';
import { Grid, Box, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions, Alert } from '@mui/material';
import { Add } from '@mui/icons-material';
import TargetForm from '../components/targets/TargetForm';
import TargetTable from '../components/targets/TargetTable';
import StatusTimeline from '../components/targets/StatusTimeline';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchTargets, createTarget, selectTargets, selectTargetLoading, selectTargetSaving } from '../store/slices/targetSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

/* Demo targets */
const DEMO_TARGETS = [
  { id: '1', organization_id: 'org_default', target_name: 'Near-term Scope 1+2 Absolute', target_type: 'near_term' as const, scope_coverage: ['scope_1', 'scope_2'] as string[], reduction_type: 'absolute' as const, method: 'aca' as const, base_year: 2019, target_year: 2030, base_emissions: 90000, target_emissions: 52200, reduction_pct: 42, alignment: '1.5C' as const, status: 'validated' as const, scope_details: [], created_at: '2024-06-15', updated_at: '2025-01-10' },
  { id: '2', organization_id: 'org_default', target_name: 'Near-term Scope 3 Absolute', target_type: 'near_term' as const, scope_coverage: ['scope_3'] as string[], reduction_type: 'absolute' as const, method: 'aca' as const, base_year: 2019, target_year: 2030, base_emissions: 210000, target_emissions: 155400, reduction_pct: 26, alignment: 'WB2C' as const, status: 'submitted' as const, scope_details: [], created_at: '2024-09-01', updated_at: '2025-02-20' },
  { id: '3', organization_id: 'org_default', target_name: 'Long-term Net-Zero S1+2', target_type: 'long_term' as const, scope_coverage: ['scope_1', 'scope_2'] as string[], reduction_type: 'absolute' as const, method: 'aca' as const, base_year: 2019, target_year: 2050, base_emissions: 90000, target_emissions: 9000, reduction_pct: 90, alignment: '1.5C' as const, status: 'active' as const, scope_details: [], created_at: '2024-06-15', updated_at: '2025-01-10' },
  { id: '4', organization_id: 'org_default', target_name: 'SDA Power Sector Intensity', target_type: 'near_term' as const, scope_coverage: ['scope_1'] as string[], reduction_type: 'intensity' as const, method: 'sda' as const, base_year: 2019, target_year: 2030, base_emissions: 0.45, target_emissions: 0.18, reduction_pct: 60, alignment: '1.5C' as const, status: 'draft' as const, scope_details: [], created_at: '2025-01-05', updated_at: '2025-02-28' },
];

const TargetConfiguration: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const storeTargets = useAppSelector(selectTargets);
  const loading = useAppSelector(selectTargetLoading);
  const saving = useAppSelector(selectTargetSaving);
  const [formOpen, setFormOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    dispatch(fetchTargets(orgId));
  }, [dispatch, orgId]);

  const targets = storeTargets.length > 0 ? storeTargets : DEMO_TARGETS;
  const selectedTarget = targets.find((t) => t.id === selectedId) || null;

  const handleCreate = (data: Record<string, unknown>) => {
    dispatch(createTarget(data as any));
    setFormOpen(false);
  };

  if (loading && storeTargets.length === 0) return <LoadingSpinner message="Loading targets..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Target Configuration</Typography>
          <Typography variant="body2" color="text.secondary">
            Define and manage SBTi-aligned emission reduction targets
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<Add />} onClick={() => setFormOpen(true)}>
          New Target
        </Button>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Near-term targets (5-10 years) require minimum 4.2% annual linear reduction for 1.5C alignment.
        Long-term targets must achieve at least 90% reduction by 2050 or sooner.
      </Alert>

      {/* Target Table */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TargetTable
            targets={targets as any}
            onSelect={(id) => setSelectedId(id)}
            onEdit={(id) => { setSelectedId(id); setFormOpen(true); }}
          />
        </Grid>

        {/* Status Timeline for selected target */}
        {selectedTarget && (
          <Grid item xs={12}>
            <StatusTimeline target={selectedTarget as any} />
          </Grid>
        )}
      </Grid>

      {/* Target Form Dialog */}
      <Dialog open={formOpen} onClose={() => setFormOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{selectedId ? 'Edit Target' : 'Create New Target'}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TargetForm
              target={selectedId ? (selectedTarget as any) : undefined}
              onSubmit={handleCreate}
              saving={saving}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setFormOpen(false); setSelectedId(null); }}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TargetConfiguration;
