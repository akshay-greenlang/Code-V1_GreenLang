/**
 * ScoreCard - Score overview card with band indicator
 *
 * Displays the predicted CDP score, level, band, and delta
 * from the previous submission year.
 */

import React from 'react';
import { Card, CardContent, Box, Typography, Chip } from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import type { ScoringLevel, ScoringBand } from '../../types';
import { getScoringLevelColor } from '../../utils/formatters';
import { getBandLabel } from '../../utils/scoringHelpers';

interface ScoreCardProps {
  score: number;
  level: ScoringLevel;
  band: ScoringBand;
  previousScore: number | null;
  previousLevel: ScoringLevel | null;
}

const ScoreCard: React.FC<ScoreCardProps> = ({
  score,
  level,
  band,
  previousScore,
  previousLevel,
}) => {
  const color = getScoringLevelColor(level);
  const delta = previousScore != null ? score - previousScore : null;

  return (
    <Card sx={{ borderLeft: `4px solid ${color}` }}>
      <CardContent>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
          Predicted CDP Score
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mt: 0.5 }}>
          <Typography variant="h3" fontWeight={800} sx={{ color }}>
            {level}
          </Typography>
          <Typography variant="h6" color="text.secondary">
            {score.toFixed(0)}%
          </Typography>
        </Box>
        <Chip
          label={band}
          size="small"
          sx={{
            mt: 1,
            backgroundColor: color + '15',
            color,
            fontWeight: 600,
          }}
        />
        {delta != null && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
            {delta > 0 ? (
              <TrendingUp fontSize="small" sx={{ color: '#2e7d32' }} />
            ) : delta < 0 ? (
              <TrendingDown fontSize="small" sx={{ color: '#c62828' }} />
            ) : (
              <TrendingFlat fontSize="small" sx={{ color: 'text.secondary' }} />
            )}
            <Typography
              variant="caption"
              color={delta > 0 ? 'success.main' : delta < 0 ? 'error.main' : 'text.secondary'}
              fontWeight={600}
            >
              {delta > 0 ? '+' : ''}{delta.toFixed(1)} pts vs {previousLevel || 'prior year'}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ScoreCard;
