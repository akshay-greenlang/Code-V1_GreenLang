/**
 * QualityScore - Data quality gauge visualization
 *
 * Renders a circular progress gauge (0-100%) with color coding:
 * green (>80%), yellow (60-80%), red (<60%). Below the gauge,
 * shows a breakdown by quality dimension (completeness, accuracy,
 * consistency, timeliness, methodology).
 */

import React from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  LinearProgress,
  Card,
  CardContent,
} from '@mui/material';
import type { QualityDimensions } from '../../types';
import { getQualityColor, getQualityLabel } from '../../utils/formatters';

interface QualityScoreProps {
  score: number;
  dimensions?: QualityDimensions;
}

const DIMENSION_LABELS: Record<keyof QualityDimensions, string> = {
  completeness: 'Completeness',
  accuracy: 'Accuracy',
  consistency: 'Consistency',
  timeliness: 'Timeliness',
  methodology: 'Methodology',
};

const getColorHex = (score: number): string => {
  if (score >= 80) return '#2e7d32';
  if (score >= 60) return '#ef6c00';
  return '#c62828';
};

const QualityScore: React.FC<QualityScoreProps> = ({ score, dimensions }) => {
  const colorHex = getColorHex(score);
  const qualityLabel = getQualityLabel(score);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Data Quality
        </Typography>

        {/* Circular gauge */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            position: 'relative',
            my: 2,
          }}
        >
          <CircularProgress
            variant="determinate"
            value={100}
            size={120}
            thickness={6}
            sx={{ color: '#e0e0e0', position: 'absolute' }}
          />
          <CircularProgress
            variant="determinate"
            value={score}
            size={120}
            thickness={6}
            sx={{ color: colorHex }}
          />
          <Box
            sx={{
              position: 'absolute',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Typography variant="h4" sx={{ fontWeight: 700, color: colorHex }}>
              {score.toFixed(0)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {qualityLabel}
            </Typography>
          </Box>
        </Box>

        {/* Dimension breakdown */}
        {dimensions && (
          <Box sx={{ mt: 2 }}>
            {(Object.keys(DIMENSION_LABELS) as Array<keyof QualityDimensions>).map(
              (key) => {
                const value = dimensions[key];
                const dimColor = getColorHex(value);
                return (
                  <Box key={key} sx={{ mb: 1.5 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        mb: 0.25,
                      }}
                    >
                      <Typography variant="body2" color="text.secondary">
                        {DIMENSION_LABELS[key]}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {value.toFixed(0)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={value}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: dimColor,
                          borderRadius: 3,
                        },
                      }}
                    />
                  </Box>
                );
              }
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default QualityScore;
