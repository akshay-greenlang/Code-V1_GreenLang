/**
 * AlignmentWorkflow - Step-by-step alignment, status board, funnel.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import AlignmentStepper from '../components/alignment/AlignmentStepper';
import ActivityStatusBoard from '../components/alignment/ActivityStatusBoard';
import PortfolioView from '../components/alignment/PortfolioView';
import AlignmentFunnel from '../components/alignment/AlignmentFunnel';

const AlignmentWorkflow: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Alignment Workflow
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <AlignmentStepper />
      </Grid>
      <Grid item xs={12} md={6}>
        <AlignmentFunnel />
      </Grid>
    </Grid>

    <Box sx={{ mb: 3 }}>
      <PortfolioView />
    </Box>

    <Box>
      <ActivityStatusBoard />
    </Box>
  </Box>
);

export default AlignmentWorkflow;
