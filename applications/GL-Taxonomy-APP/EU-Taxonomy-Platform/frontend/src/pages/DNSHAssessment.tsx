/**
 * DNSHAssessment - DNSH matrix, climate risk wizard, per-objective assessments.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import ObjectiveMatrix from '../components/dnsh/ObjectiveMatrix';
import ClimateRiskWizard from '../components/dnsh/ClimateRiskWizard';
import WaterAssessment from '../components/dnsh/WaterAssessment';
import CircularAssessment from '../components/dnsh/CircularAssessment';
import PollutionCheck from '../components/dnsh/PollutionCheck';
import BiodiversityCheck from '../components/dnsh/BiodiversityCheck';

const DNSHAssessment: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Do No Significant Harm (DNSH) Assessment
    </Typography>

    <Box sx={{ mb: 3 }}>
      <ObjectiveMatrix />
    </Box>

    <Box sx={{ mb: 3 }}>
      <ClimateRiskWizard />
    </Box>

    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <WaterAssessment />
      </Grid>
      <Grid item xs={12} md={6}>
        <CircularAssessment />
      </Grid>
      <Grid item xs={12} md={6}>
        <PollutionCheck />
      </Grid>
      <Grid item xs={12} md={6}>
        <BiodiversityCheck />
      </Grid>
    </Grid>
  </Box>
);

export default DNSHAssessment;
