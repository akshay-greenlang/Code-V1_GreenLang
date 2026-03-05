/**
 * ActivityScreening - NACE search, activity catalog, batch screening, eligibility results.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import NACESearch from '../components/activities/NACESearch';
import SectorFilter from '../components/activities/SectorFilter';
import BatchScreener from '../components/screening/BatchScreener';
import EligibilityResults from '../components/screening/EligibilityResults';
import ScreeningSummary from '../components/screening/ScreeningSummary';
import DeMinimisFilter from '../components/screening/DeMinimisFilter';
import ActivityCatalog from '../components/activities/ActivityCatalog';
import EligibilityMatrix from '../components/activities/EligibilityMatrix';

const ActivityScreening: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Activity Screening
    </Typography>

    <Box sx={{ mb: 3 }}>
      <NACESearch />
    </Box>

    <Box sx={{ mb: 3 }}>
      <SectorFilter />
    </Box>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={8}>
        <BatchScreener />
      </Grid>
      <Grid item xs={12} md={4}>
        <DeMinimisFilter />
      </Grid>
    </Grid>

    <Box sx={{ mb: 3 }}>
      <ScreeningSummary />
    </Box>

    <Box sx={{ mb: 3 }}>
      <EligibilityResults />
    </Box>

    <Grid container spacing={3}>
      <Grid item xs={12} md={7}>
        <ActivityCatalog />
      </Grid>
      <Grid item xs={12} md={5}>
        <EligibilityMatrix />
      </Grid>
    </Grid>
  </Box>
);

export default ActivityScreening;
