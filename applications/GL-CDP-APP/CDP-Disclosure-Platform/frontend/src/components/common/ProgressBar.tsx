/**
 * ProgressBar - Module completion progress bar
 *
 * Displays a horizontal bar with completion percentage and
 * optional label. Color transitions from red to green based on completion.
 */

import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface ProgressBarProps {
  label: string;
  value: number;
  max?: number;
  showPercentage?: boolean;
  height?: number;
  color?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  label,
  value,
  max = 100,
  showPercentage = true,
  height = 8,
  color,
}) => {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;

  const getColor = (): string => {
    if (color) return color;
    if (pct >= 80) return '#2e7d32';
    if (pct >= 60) return '#1565c0';
    if (pct >= 40) return '#ef6c00';
    return '#c62828';
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="body2" color="text.primary" sx={{ fontWeight: 500 }}>
          {label}
        </Typography>
        {showPercentage && (
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
            {pct.toFixed(0)}%
          </Typography>
        )}
      </Box>
      <LinearProgress
        variant="determinate"
        value={pct}
        sx={{
          height,
          borderRadius: height / 2,
          backgroundColor: '#e0e0e0',
          '& .MuiLinearProgress-bar': {
            borderRadius: height / 2,
            backgroundColor: getColor(),
          },
        }}
      />
    </Box>
  );
};

export default ProgressBar;
