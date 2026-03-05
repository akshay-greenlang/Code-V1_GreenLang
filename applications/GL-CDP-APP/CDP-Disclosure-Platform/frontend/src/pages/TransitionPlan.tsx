/**
 * TransitionPlan Page - Transition plan builder and tracker
 *
 * Composes PathwayChart, MilestoneTimeline, TechLeverBreakdown,
 * InvestmentPlan, and SBTiStatus for comprehensive transition planning.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import { Add, Timeline } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchTransitionPlan,
  createTransitionPlan,
  addMilestone,
  fetchPathway,
  checkSBTi,
} from '../store/slices/transitionPlanSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PathwayChart from '../components/transition/PathwayChart';
import MilestoneTimeline from '../components/transition/MilestoneTimeline';
import TechLeverBreakdown from '../components/transition/TechLeverBreakdown';
import InvestmentPlan from '../components/transition/InvestmentPlan';
import SBTiStatus from '../components/transition/SBTiStatus';

const DEMO_ORG_ID = 'demo-org';

const TransitionPlanPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { plan, pathway, loading, error } = useAppSelector(
    (s) => s.transitionPlan,
  );

  const [milestoneDialogOpen, setMilestoneDialogOpen] = useState(false);
  const [newMilestone, setNewMilestone] = useState({
    title: '',
    description: '',
    target_year: new Date().getFullYear() + 5,
    target_reduction_pct: 30,
    responsible: '',
  });

  useEffect(() => {
    dispatch(fetchTransitionPlan(DEMO_ORG_ID));
  }, [dispatch]);

  useEffect(() => {
    if (plan) {
      dispatch(fetchPathway(plan.id));
      dispatch(checkSBTi(plan.id));
    }
  }, [dispatch, plan]);

  const handleAddMilestone = () => {
    if (!plan) return;
    dispatch(addMilestone({
      planId: plan.id,
      data: newMilestone,
    }));
    setMilestoneDialogOpen(false);
    setNewMilestone({
      title: '',
      description: '',
      target_year: new Date().getFullYear() + 5,
      target_reduction_pct: 30,
      responsible: '',
    });
  };

  if (loading && !plan) return <LoadingSpinner message="Loading transition plan..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  if (!plan) {
    return (
      <Box>
        <Typography variant="h5" fontWeight={700} gutterBottom>
          Transition Plan
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          No transition plan found. Create one to get started.
        </Alert>
        <Button
          variant="contained"
          startIcon={<Timeline />}
          onClick={() => dispatch(createTransitionPlan({
            orgId: DEMO_ORG_ID,
            data: {
              title: '1.5C Transition Plan',
              target_year: 2050,
              base_year: 2020,
              base_year_emissions: 100000,
              target_emissions: 5000,
            },
          }))}
        >
          Create Transition Plan
        </Button>
      </Box>
    );
  }

  const totalReduction = plan.base_year_emissions - plan.target_emissions;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            {plan.title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {plan.base_year} to {plan.target_year} | {plan.pathway_type} pathway
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Add />}
          onClick={() => setMilestoneDialogOpen(true)}
        >
          Add Milestone
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Pathway chart */}
        <Grid item xs={12}>
          <PathwayChart
            pathway={pathway}
            baseYear={plan.base_year}
            targetYear={plan.target_year}
            baseYearEmissions={plan.base_year_emissions}
            targetEmissions={plan.target_emissions}
          />
        </Grid>

        {/* SBTi status */}
        <Grid item xs={12} md={4}>
          <SBTiStatus
            sbtiAligned={plan.sbti_aligned}
            sbtiStatus={plan.sbti_status}
            pathwayType={plan.pathway_type}
            reductionTargetPct={plan.reduction_target_pct}
            annualReductionRate={plan.annual_reduction_rate}
            boardOversight={plan.board_oversight}
            publiclyDisclosed={plan.publicly_disclosed}
            targetYear={plan.target_year}
          />
        </Grid>

        {/* Investment plan */}
        <Grid item xs={12} md={8}>
          <InvestmentPlan
            levers={plan.technology_levers}
            totalInvestment={plan.investment_total_usd}
            lowCarbonRevenuePct={plan.low_carbon_revenue_pct}
          />
        </Grid>

        {/* Milestone timeline */}
        <Grid item xs={12} md={5}>
          <MilestoneTimeline milestones={plan.milestones} />
        </Grid>

        {/* Tech lever breakdown */}
        <Grid item xs={12} md={7}>
          <TechLeverBreakdown
            levers={plan.technology_levers}
            totalReduction={totalReduction}
          />
        </Grid>
      </Grid>

      {/* Add milestone dialog */}
      <Dialog
        open={milestoneDialogOpen}
        onClose={() => setMilestoneDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add Milestone</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Milestone Title"
              value={newMilestone.title}
              onChange={(e) => setNewMilestone({ ...newMilestone, title: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={newMilestone.description}
              onChange={(e) => setNewMilestone({ ...newMilestone, description: e.target.value })}
              fullWidth
              multiline
              rows={2}
            />
            <TextField
              label="Target Year"
              type="number"
              value={newMilestone.target_year}
              onChange={(e) => setNewMilestone({ ...newMilestone, target_year: parseInt(e.target.value) })}
              fullWidth
              required
            />
            <TextField
              label="Target Reduction (%)"
              type="number"
              value={newMilestone.target_reduction_pct}
              onChange={(e) => setNewMilestone({ ...newMilestone, target_reduction_pct: parseFloat(e.target.value) })}
              fullWidth
              required
            />
            <TextField
              label="Responsible Person"
              value={newMilestone.responsible}
              onChange={(e) => setNewMilestone({ ...newMilestone, responsible: e.target.value })}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMilestoneDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddMilestone}
            disabled={!newMilestone.title}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TransitionPlanPage;
