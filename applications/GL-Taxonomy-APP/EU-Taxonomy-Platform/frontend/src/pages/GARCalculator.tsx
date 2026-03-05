/**
 * GARCalculator - GAR stock/flow, BTAR, exposure classifier, EBA template.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import GARStockCard from '../components/gar/GARStockCard';
import GARFlowCard from '../components/gar/GARFlowCard';
import BTARCalculator from '../components/gar/BTARCalculator';
import ExposureClassifier from '../components/gar/ExposureClassifier';
import EBATemplatePreview from '../components/gar/EBATemplatePreview';

const GARCalculator: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      GAR Calculator
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={4}>
        <GARStockCard />
      </Grid>
      <Grid item xs={12} md={4}>
        <GARFlowCard />
      </Grid>
      <Grid item xs={12} md={4}>
        <BTARCalculator />
      </Grid>
    </Grid>

    <Box sx={{ mb: 3 }}>
      <ExposureClassifier />
    </Box>

    <Box>
      <EBATemplatePreview />
    </Box>
  </Box>
);

export default GARCalculator;
