/**
 * SubstantialContribution - SC assessment with criteria, thresholds, evidence.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import ObjectiveSelector from '../components/substantialContribution/ObjectiveSelector';
import TSCChecklist from '../components/substantialContribution/TSCChecklist';
import ThresholdEvaluator from '../components/substantialContribution/ThresholdEvaluator';
import EvidencePanel from '../components/substantialContribution/EvidencePanel';
import SCResultCard from '../components/substantialContribution/SCResultCard';
import ActivityDetail from '../components/activities/ActivityDetail';

const SubstantialContribution: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Substantial Contribution Assessment
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <ActivityDetail />
      </Grid>
      <Grid item xs={12} md={6}>
        <ObjectiveSelector />
      </Grid>
    </Grid>

    <Box sx={{ mb: 3 }}>
      <SCResultCard />
    </Box>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <TSCChecklist />
      </Grid>
      <Grid item xs={12} md={6}>
        <ThresholdEvaluator />
      </Grid>
    </Grid>

    <Box>
      <EvidencePanel />
    </Box>
  </Box>
);

export default SubstantialContribution;
