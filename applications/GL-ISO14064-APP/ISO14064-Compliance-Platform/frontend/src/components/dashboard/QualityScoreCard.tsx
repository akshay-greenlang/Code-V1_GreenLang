/**
 * GL-ISO14064-APP v1.0 - Data Quality Score Card
 *
 * Circular gauge showing overall data quality score (0-100) with
 * colour-coded thresholds per ISO 14064-1 Clause 7 requirements.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress, useTheme } from '@mui/material';

interface Props {
  score: number;
  completeness: number;
  label?: string;
}

function scoreColor(score: number): string {
  if (score >= 80) return '#2e7d32';
  if (score >= 60) return '#ef6c00';
  return '#e53935';
}

function scoreLabel(score: number): string {
  if (score >= 80) return 'High';
  if (score >= 60) return 'Medium';
  return 'Low';
}

const QualityScoreCard: React.FC<Props> = ({ score, completeness, label = 'Data Quality' }) => {
  const theme = useTheme();
  const clampedScore = Math.min(100, Math.max(0, score));
  const color = scoreColor(clampedScore);

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {label}
        </Typography>
        <Box display="flex" alignItems="center" gap={3} mt={1}>
          <Box position="relative" display="inline-flex">
            <CircularProgress
              variant="determinate"
              value={100}
              size={96}
              thickness={6}
              sx={{ color: theme.palette.grey[200], position: 'absolute' }}
            />
            <CircularProgress
              variant="determinate"
              value={clampedScore}
              size={96}
              thickness={6}
              sx={{ color }}
            />
            <Box
              position="absolute"
              top={0}
              left={0}
              bottom={0}
              right={0}
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
            >
              <Typography variant="h6" fontWeight={700} sx={{ color, lineHeight: 1 }}>
                {clampedScore}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                / 100
              </Typography>
            </Box>
          </Box>
          <Box flex={1}>
            <Typography variant="body2" fontWeight={600} sx={{ color }}>
              {scoreLabel(clampedScore)} Quality
            </Typography>
            <Typography variant="body2" color="text.secondary" mt={0.5}>
              Completeness: {completeness.toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Clause 7 compliance indicator
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default QualityScoreCard;
