/**
 * ProgressBar - Horizontal progress bar with label and percentage.
 */

import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface ProgressBarProps {
  value: number;
  maxValue?: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  height?: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  maxValue = 100,
  label,
  showPercentage = true,
  color = 'primary',
  height = 8,
}) => {
  const percent = Math.min((value / maxValue) * 100, 100);

  return (
    <Box sx={{ width: '100%' }}>
      {(label || showPercentage) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          {label && (
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {label}
            </Typography>
          )}
          {showPercentage && (
            <Typography variant="body2" color="text.secondary">
              {percent.toFixed(1)}%
            </Typography>
          )}
        </Box>
      )}
      <LinearProgress
        variant="determinate"
        value={percent}
        color={color}
        sx={{ height, borderRadius: height / 2 }}
      />
    </Box>
  );
};

export default ProgressBar;
