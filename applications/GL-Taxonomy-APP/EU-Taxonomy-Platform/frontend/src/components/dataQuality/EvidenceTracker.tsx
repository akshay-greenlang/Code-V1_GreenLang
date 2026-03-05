/**
 * EvidenceTracker - Track evidence coverage for taxonomy assessments.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Grid } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import ProgressBar from '../common/ProgressBar';

const DEMO = {
  total_required: 120,
  total_provided: 92,
  coverage_pct: 76.7,
  by_category: [
    { category: 'SC Criteria', required: 45, provided: 38 },
    { category: 'DNSH Assessment', required: 35, provided: 28 },
    { category: 'Safeguard Due Diligence', required: 20, provided: 14 },
    { category: 'Climate Risk', required: 12, provided: 8 },
    { category: 'Financial Data', required: 8, provided: 4 },
  ],
};

const EvidenceTracker: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Evidence Coverage</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Box sx={{ textAlign: 'center' }}>
            <ScoreGauge value={DEMO.coverage_pct} label="Coverage" size={100} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {DEMO.total_provided}/{DEMO.total_required} documents provided
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={8}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {DEMO.by_category.map(cat => (
              <ProgressBar
                key={cat.category}
                value={cat.provided}
                maxValue={cat.required}
                label={`${cat.category} (${cat.provided}/${cat.required})`}
                color={cat.provided / cat.required >= 0.8 ? 'success' : 'warning'}
              />
            ))}
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default EvidenceTracker;
