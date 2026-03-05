/**
 * CoverageGauge - S1+2 and S3 coverage gauges.
 */
import React from 'react';
import { Card, CardContent, Typography, Grid, Box } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';

interface CoverageGaugeProps { scope12Coverage: number; scope3Coverage: number; }

const CoverageGauge: React.FC<CoverageGaugeProps> = ({ scope12Coverage, scope3Coverage }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Scope Coverage</Typography>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <ScoreGauge value={scope12Coverage} label="Scope 1+2" subtitle="Min 95% required" size={120} color={scope12Coverage >= 95 ? '#2E7D32' : '#C62828'} />
        </Grid>
        <Grid item xs={6}>
          <ScoreGauge value={scope3Coverage} label="Scope 3" subtitle="Min 67% required" size={120} color={scope3Coverage >= 67 ? '#2E7D32' : scope3Coverage >= 40 ? '#EF6C00' : '#C62828'} />
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default CoverageGauge;
