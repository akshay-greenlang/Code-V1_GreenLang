/**
 * ScreeningSummary - Summary statistics of the screening process.
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, Box } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';

const DEMO = {
  total_activities: 53,
  eligible_count: 38,
  not_eligible_count: 15,
  turnover_eligible_pct: 68.2,
  capex_eligible_pct: 72.1,
  opex_eligible_pct: 55.4,
};

interface ScreeningSummaryProps {
  data?: typeof DEMO;
}

const ScreeningSummary: React.FC<ScreeningSummaryProps> = ({ data = DEMO }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Screening Summary
      </Typography>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} md={3}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main' }}>
              {data.eligible_count}/{data.total_activities}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Activities Eligible
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={4} md={3}>
          <ScoreGauge value={data.turnover_eligible_pct} label="Turnover" color="#1B5E20" />
        </Grid>
        <Grid item xs={4} md={3}>
          <ScoreGauge value={data.capex_eligible_pct} label="CapEx" color="#0D47A1" />
        </Grid>
        <Grid item xs={4} md={3}>
          <ScoreGauge value={data.opex_eligible_pct} label="OpEx" color="#E65100" />
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default ScreeningSummary;
