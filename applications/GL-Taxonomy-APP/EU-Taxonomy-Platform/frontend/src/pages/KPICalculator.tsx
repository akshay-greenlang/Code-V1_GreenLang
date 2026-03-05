/**
 * KPICalculator - Three KPI cards, objective breakdown, double-counting guard.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import TurnoverKPICard from '../components/kpi/TurnoverKPICard';
import CapExKPICard from '../components/kpi/CapExKPICard';
import OpExKPICard from '../components/kpi/OpExKPICard';
import ObjectiveBreakdownChart from '../components/kpi/ObjectiveBreakdownChart';
import DoubleCountingGuard from '../components/kpi/DoubleCountingGuard';

const KPICalculator: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      KPI Calculator
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={4}>
        <TurnoverKPICard />
      </Grid>
      <Grid item xs={12} md={4}>
        <CapExKPICard />
      </Grid>
      <Grid item xs={12} md={4}>
        <OpExKPICard />
      </Grid>
    </Grid>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={8}>
        <ObjectiveBreakdownChart />
      </Grid>
      <Grid item xs={12} md={4}>
        <DoubleCountingGuard />
      </Grid>
    </Grid>
  </Box>
);

export default KPICalculator;
