/**
 * RegulatoryUpdates - DA timeline, TSC changes, Omnibus impact.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import DAVersionTracker from '../components/regulatory/DAVersionTracker';
import TSCChangeLog from '../components/regulatory/TSCChangeLog';
import OmnibusTimeline from '../components/regulatory/OmnibusTimeline';

const RegulatoryUpdates: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Regulatory Updates
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <DAVersionTracker />
      </Grid>
      <Grid item xs={12} md={6}>
        <OmnibusTimeline />
      </Grid>
    </Grid>

    <Box>
      <TSCChangeLog />
    </Box>
  </Box>
);

export default RegulatoryUpdates;
