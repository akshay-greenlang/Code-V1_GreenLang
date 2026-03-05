/**
 * MinimumSafeguards - 4-topic safeguard assessment with checklists and trackers.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import TopicAssessmentCard from '../components/safeguards/TopicAssessmentCard';
import ProceduralChecklist from '../components/safeguards/ProceduralChecklist';
import OutcomeMonitor from '../components/safeguards/OutcomeMonitor';
import DueDiligenceTracker from '../components/safeguards/DueDiligenceTracker';

const MinimumSafeguards: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Minimum Safeguards Assessment
    </Typography>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} sm={6} md={3}>
        <TopicAssessmentCard topic="Human Rights" proceduralScore={85} outcomeScore={90} pass frameworks={['UNGP', 'OECD Guidelines', 'ILO Core']} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <TopicAssessmentCard topic="Anti-Corruption" proceduralScore={90} outcomeScore={95} pass frameworks={['UNCAC', 'OECD Anti-Bribery']} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <TopicAssessmentCard topic="Taxation" proceduralScore={80} outcomeScore={85} pass frameworks={['OECD Tax Guidelines']} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <TopicAssessmentCard topic="Fair Competition" proceduralScore={75} outcomeScore={80} pass frameworks={['EU Competition Law']} />
      </Grid>
    </Grid>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={6}>
        <ProceduralChecklist />
      </Grid>
      <Grid item xs={12} md={6}>
        <OutcomeMonitor />
      </Grid>
    </Grid>

    <Box>
      <DueDiligenceTracker />
    </Box>
  </Box>
);

export default MinimumSafeguards;
