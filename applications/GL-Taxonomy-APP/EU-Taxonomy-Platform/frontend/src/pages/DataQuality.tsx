/**
 * DataQuality - 5-dimension scorecard, completeness, evidence, improvement actions.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import QualityScorecard from '../components/dataQuality/QualityScorecard';
import CompletenessMatrix from '../components/dataQuality/CompletenessMatrix';
import EvidenceTracker from '../components/dataQuality/EvidenceTracker';
import ImprovementActions from '../components/dataQuality/ImprovementActions';

const DataQuality: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Data Quality
    </Typography>

    <Box sx={{ mb: 3 }}>
      <QualityScorecard />
    </Box>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <CompletenessMatrix />
      </Grid>
      <Grid item xs={12} md={6}>
        <EvidenceTracker />
      </Grid>
    </Grid>

    <Box>
      <ImprovementActions />
    </Box>
  </Box>
);

export default DataQuality;
