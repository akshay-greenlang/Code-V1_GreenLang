/**
 * GapAnalysis - Gap heatmap, action plan, priority matrix.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import GapHeatmap from '../components/gap/GapHeatmap';
import ActionPlanTable from '../components/gap/ActionPlanTable';
import PriorityMatrix from '../components/gap/PriorityMatrix';

const GapAnalysis: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Gap Analysis
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <GapHeatmap />
      </Grid>
      <Grid item xs={12} md={6}>
        <PriorityMatrix />
      </Grid>
    </Grid>

    <Box>
      <ActionPlanTable />
    </Box>
  </Box>
);

export default GapAnalysis;
