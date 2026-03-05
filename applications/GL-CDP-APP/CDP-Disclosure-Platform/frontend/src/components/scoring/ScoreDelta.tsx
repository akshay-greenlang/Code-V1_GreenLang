/**
 * ScoreDelta - Before/after score comparison
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { ArrowForward, TrendingUp } from '@mui/icons-material';
import type { ScoringLevel } from '../../types';
import { getScoringLevelColor } from '../../utils/formatters';

interface ScoreDeltaProps {
  currentScore: number;
  currentLevel: ScoringLevel;
  projectedScore: number;
  projectedLevel: ScoringLevel;
}

const ScoreDelta: React.FC<ScoreDeltaProps> = ({ currentScore, currentLevel, projectedScore, projectedLevel }) => {
  const delta = projectedScore - currentScore;
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Score Impact</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Current</Typography>
            <Typography variant="h4" fontWeight={700} sx={{ color: getScoringLevelColor(currentLevel) }}>
              {currentLevel}
            </Typography>
            <Typography variant="body2">{currentScore.toFixed(0)}%</Typography>
          </Box>
          <ArrowForward sx={{ fontSize: 32, color: '#9e9e9e' }} />
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Projected</Typography>
            <Typography variant="h4" fontWeight={700} sx={{ color: getScoringLevelColor(projectedLevel) }}>
              {projectedLevel}
            </Typography>
            <Typography variant="body2">{projectedScore.toFixed(0)}%</Typography>
          </Box>
        </Box>
        {delta > 0 && (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5, mt: 2 }}>
            <TrendingUp sx={{ color: '#2e7d32' }} />
            <Typography variant="h6" color="success.main" fontWeight={700}>
              +{delta.toFixed(1)} points
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ScoreDelta;
