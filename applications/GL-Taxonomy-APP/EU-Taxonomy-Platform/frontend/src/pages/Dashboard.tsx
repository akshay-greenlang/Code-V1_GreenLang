/**
 * Dashboard - Executive overview with KPI summaries, alignment, sectors, trends, radar.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import KPISummaryCards from '../components/dashboard/KPISummaryCards';
import AlignmentOverview from '../components/dashboard/AlignmentOverview';
import SectorBreakdownChart from '../components/dashboard/SectorBreakdownChart';
import EligibleVsAligned from '../components/dashboard/EligibleVsAligned';
import TimelineTrend from '../components/dashboard/TimelineTrend';
import ObjectiveRadar from '../components/dashboard/ObjectiveRadar';

const Dashboard: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      EU Taxonomy Dashboard
    </Typography>

    <Box sx={{ mb: 3 }}>
      <KPISummaryCards />
    </Box>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <AlignmentOverview />
      </Grid>
      <Grid item xs={12} md={6}>
        <EligibleVsAligned />
      </Grid>
    </Grid>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <SectorBreakdownChart />
      </Grid>
      <Grid item xs={12} md={6}>
        <ObjectiveRadar />
      </Grid>
    </Grid>

    <Box>
      <TimelineTrend />
    </Box>
  </Box>
);

export default Dashboard;
