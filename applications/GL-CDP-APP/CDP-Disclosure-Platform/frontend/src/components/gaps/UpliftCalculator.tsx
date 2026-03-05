/**
 * UpliftCalculator - Score uplift prediction
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';
import type { GapItem } from '../../types';

interface UpliftCalculatorProps { gaps: GapItem[]; currentScore: number; }

const UpliftCalculator: React.FC<UpliftCalculatorProps> = ({ gaps, currentScore }) => {
  const unresolvedGaps = gaps.filter((g) => !g.is_resolved);
  const totalUplift = unresolvedGaps.reduce((sum, g) => sum + g.uplift_points, 0);
  const projectedScore = Math.min(100, currentScore + totalUplift);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Uplift Potential</Typography>
        <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Current</Typography>
            <Typography variant="h5" fontWeight={700}>{currentScore.toFixed(0)}%</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <TrendingUp sx={{ color: '#2e7d32' }} />
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Projected</Typography>
            <Typography variant="h5" fontWeight={700} color="success.main">{projectedScore.toFixed(0)}%</Typography>
          </Box>
        </Box>
        <Chip label={`+${totalUplift.toFixed(1)} pts from ${unresolvedGaps.length} gaps`} color="success" variant="outlined" />
      </CardContent>
    </Card>
  );
};

export default UpliftCalculator;
