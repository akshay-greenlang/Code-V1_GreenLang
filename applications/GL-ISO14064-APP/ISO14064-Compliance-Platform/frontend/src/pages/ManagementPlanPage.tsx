/**
 * ManagementPlanPage - GHG management plan with actions CRUD,
 * impact projection, and quality management.
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Alert,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Card,
  CardContent,
} from '@mui/material';
import { Add } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchPlan,
  fetchActions as fetchMgmtActions,
  addAction,
  deleteAction,
  fetchQualityPlan,
} from '../store/slices/managementSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ActionPlanTable from '../components/management/ActionPlanTable';
import ImpactProjectionChart from '../components/management/ImpactProjectionChart';
import DataQualityScorecard from '../components/quality/DataQualityScorecard';
import QualityProcedureList from '../components/quality/QualityProcedureList';
import type { ManagementAction, CreateManagementActionRequest } from '../types';
import {
  ActionCategory,
  ActionStatus,
  ISOCategory,
  ISO_CATEGORY_SHORT_NAMES,
} from '../types';

const DEMO_ORG_ID = 'demo-org';
const REPORTING_YEAR = new Date().getFullYear() - 1;

const ManagementPlanPage: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { plan, actions, qualityPlan, loading, error } = useSelector(
    (s: AppRootState) => s.management,
  );
  const dataQuality = useSelector((s: AppRootState) => s.emissions.dataQuality);

  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newAction, setNewAction] = useState<CreateManagementActionRequest>({
    title: '',
    description: '',
    action_category: ActionCategory.EMISSION_REDUCTION,
    target_category: null,
    priority: 'medium',
    estimated_reduction_tco2e: null,
    estimated_cost_usd: null,
    responsible_person: '',
    target_date: null,
  });

  useEffect(() => {
    dispatch(fetchPlan({ orgId: DEMO_ORG_ID, reportingYear: REPORTING_YEAR }));
    dispatch(fetchMgmtActions(DEMO_ORG_ID));
    dispatch(fetchQualityPlan(DEMO_ORG_ID));
  }, [dispatch]);

  const handleAddAction = () => {
    dispatch(addAction({ orgId: DEMO_ORG_ID, payload: newAction }));
    setAddDialogOpen(false);
    setNewAction({
      title: '',
      description: '',
      action_category: ActionCategory.EMISSION_REDUCTION,
      target_category: null,
      priority: 'medium',
      estimated_reduction_tco2e: null,
      estimated_cost_usd: null,
      responsible_person: '',
      target_date: null,
    });
  };

  const handleDeleteAction = (actionId: string) => {
    dispatch(deleteAction({ orgId: DEMO_ORG_ID, actionId }));
  };

  const handleEditAction = (action: ManagementAction) => {
    // For now, open add dialog pre-filled (could be enhanced to edit)
    setNewAction({
      title: action.title,
      description: action.description,
      action_category: action.action_category,
      target_category: action.target_category,
      priority: action.priority,
      estimated_reduction_tco2e: action.estimated_reduction_tco2e,
      estimated_cost_usd: action.estimated_cost_usd,
      responsible_person: action.responsible_person,
      target_date: action.target_date,
    });
    setAddDialogOpen(true);
  };

  if (loading && !plan && actions.length === 0) {
    return <LoadingSpinner message="Loading management plan..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            GHG Management Plan
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ISO 14064-1 Clause 9 - Management actions and quality procedures
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setAddDialogOpen(true)}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          Add Action
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Plan summary */}
      {plan && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Typography variant="caption" color="text.secondary">Plan Objectives</Typography>
                {plan.objectives.map((obj, idx) => (
                  <Typography key={idx} variant="body2">- {obj}</Typography>
                ))}
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="caption" color="text.secondary">Total Planned Reduction</Typography>
                <Typography variant="h6" fontWeight={700} color="success.main">
                  {plan.total_planned_reduction_tco2e.toLocaleString()} tCO2e
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="caption" color="text.secondary">Total Investment</Typography>
                <Typography variant="h6" fontWeight={700}>
                  ${plan.total_planned_investment_usd.toLocaleString()}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Review cycle: {plan.review_cycle}
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Impact Projection + Actions */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={5}>
          <ImpactProjectionChart actions={actions} />
        </Grid>
        <Grid item xs={12} md={7}>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Management Actions ({actions.length})
          </Typography>
          <ActionPlanTable
            actions={actions}
            onEdit={handleEditAction}
            onDelete={handleDeleteAction}
          />
        </Grid>
      </Grid>

      {/* Quality Management */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <DataQualityScorecard quality={dataQuality} />
        </Grid>
        <Grid item xs={12} md={6}>
          {qualityPlan?.procedures ? (
            <QualityProcedureList procedures={qualityPlan.procedures} />
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  No quality procedures defined yet.
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Add Action Dialog */}
      <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add Management Action</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12} sm={8}>
              <TextField
                fullWidth
                required
                label="Title"
                value={newAction.title}
                onChange={(e) => setNewAction({ ...newAction, title: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                select
                label="Priority"
                value={newAction.priority}
                onChange={(e) => setNewAction({ ...newAction, priority: e.target.value })}
              >
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={newAction.description}
                onChange={(e) => setNewAction({ ...newAction, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Action Category"
                value={newAction.action_category}
                onChange={(e) =>
                  setNewAction({ ...newAction, action_category: e.target.value as ActionCategory })
                }
              >
                {Object.values(ActionCategory).map((c) => (
                  <MenuItem key={c} value={c}>{c.replace(/_/g, ' ')}</MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Target ISO Category"
                value={newAction.target_category || ''}
                onChange={(e) =>
                  setNewAction({
                    ...newAction,
                    target_category: (e.target.value as ISOCategory) || null,
                  })
                }
              >
                <MenuItem value="">All Categories</MenuItem>
                {Object.values(ISOCategory).map((c) => (
                  <MenuItem key={c} value={c}>{ISO_CATEGORY_SHORT_NAMES[c]}</MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                type="number"
                label="Est. Reduction (tCO2e)"
                value={newAction.estimated_reduction_tco2e ?? ''}
                onChange={(e) =>
                  setNewAction({
                    ...newAction,
                    estimated_reduction_tco2e: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                type="number"
                label="Est. Cost (USD)"
                value={newAction.estimated_cost_usd ?? ''}
                onChange={(e) =>
                  setNewAction({
                    ...newAction,
                    estimated_cost_usd: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="Responsible Person"
                value={newAction.responsible_person}
                onChange={(e) => setNewAction({ ...newAction, responsible_person: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="date"
                label="Target Date"
                value={newAction.target_date || ''}
                onChange={(e) => setNewAction({ ...newAction, target_date: e.target.value || null })}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleAddAction}
            variant="contained"
            disabled={!newAction.title}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Add Action
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ManagementPlanPage;
