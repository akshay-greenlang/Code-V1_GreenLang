/**
 * TrendArrows - Year-over-year trend direction indicators.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Grid } from '@mui/material';
import { TrendingDown, TrendingUp, TrendingFlat } from '@mui/icons-material';

interface TrendArrowsProps { yearOverYear: number; annualReductionRequired: number; actualAnnualReduction: number; }

const TrendArrows: React.FC<TrendArrowsProps> = ({ yearOverYear, annualReductionRequired, actualAnnualReduction }) => {
  const isDecreasing = yearOverYear < 0;
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Trend Analysis</Typography>
        <Grid container spacing={2}>
          <Grid item xs={4} sx={{ textAlign: 'center' }}>
            {isDecreasing ? <TrendingDown sx={{ fontSize: 36, color: '#2E7D32' }} /> : yearOverYear === 0 ? <TrendingFlat sx={{ fontSize: 36, color: '#EF6C00' }} /> : <TrendingUp sx={{ fontSize: 36, color: '#C62828' }} />}
            <Typography variant="body2" fontWeight={600}>{yearOverYear >= 0 ? '+' : ''}{yearOverYear.toFixed(1)}%</Typography>
            <Typography variant="caption" color="text.secondary">YoY Change</Typography>
          </Grid>
          <Grid item xs={4} sx={{ textAlign: 'center' }}>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#0D47A1' }}>{annualReductionRequired.toFixed(1)}%</Typography>
            <Typography variant="caption" color="text.secondary">Required Rate</Typography>
          </Grid>
          <Grid item xs={4} sx={{ textAlign: 'center' }}>
            <Typography variant="h5" sx={{ fontWeight: 700, color: actualAnnualReduction >= annualReductionRequired ? '#2E7D32' : '#C62828' }}>{actualAnnualReduction.toFixed(1)}%</Typography>
            <Typography variant="caption" color="text.secondary">Actual Rate</Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default TrendArrows;
