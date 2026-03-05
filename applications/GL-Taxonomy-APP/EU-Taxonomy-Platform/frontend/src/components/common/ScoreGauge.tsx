/**
 * ScoreGauge - Circular gauge for displaying scores and percentages.
 */

import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

interface ScoreGaugeProps {
  value: number;
  maxValue?: number;
  label: string;
  sublabel?: string;
  size?: number;
  thickness?: number;
  color?: string;
  format?: 'percent' | 'number' | 'grade';
}

const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  value,
  maxValue = 100,
  label,
  sublabel,
  size = 120,
  thickness = 4,
  color,
  format = 'percent',
}) => {
  const normalizedValue = Math.min((value / maxValue) * 100, 100);

  const getColor = () => {
    if (color) return color;
    if (normalizedValue >= 80) return '#2E7D32';
    if (normalizedValue >= 60) return '#558B2F';
    if (normalizedValue >= 40) return '#EF6C00';
    return '#C62828';
  };

  const displayValue = () => {
    switch (format) {
      case 'percent': return `${value.toFixed(1)}%`;
      case 'number': return value.toFixed(0);
      case 'grade':
        if (value >= 90) return 'A';
        if (value >= 80) return 'B';
        if (value >= 60) return 'C';
        if (value >= 40) return 'D';
        return 'F';
      default: return `${value.toFixed(1)}%`;
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          variant="determinate"
          value={100}
          size={size}
          thickness={thickness}
          sx={{ color: '#E0E0E0', position: 'absolute' }}
        />
        <CircularProgress
          variant="determinate"
          value={normalizedValue}
          size={size}
          thickness={thickness}
          sx={{ color: getColor() }}
        />
        <Box
          sx={{
            position: 'absolute',
            top: 0, left: 0, bottom: 0, right: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 700, color: getColor() }}>
            {displayValue()}
          </Typography>
        </Box>
      </Box>
      <Typography variant="body2" sx={{ fontWeight: 600, textAlign: 'center' }}>
        {label}
      </Typography>
      {sublabel && (
        <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
          {sublabel}
        </Typography>
      )}
    </Box>
  );
};

export default ScoreGauge;
