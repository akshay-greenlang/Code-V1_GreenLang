/**
 * ProgressBar - Linear progress indicator with label and value display.
 */

import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface ProgressBarProps {
  value: number;
  label: string;
  showValue?: boolean;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'success' | 'info';
  height?: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value, label, showValue = true, color, height = 8,
}) => {
  const clampedValue = Math.min(Math.max(value, 0), 100);
  const autoColor = color || (clampedValue >= 75 ? 'success' : clampedValue >= 50 ? 'warning' : 'error');

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>{label}</Typography>
        {showValue && (
          <Typography variant="body2" sx={{ fontWeight: 600 }}>{clampedValue.toFixed(0)}%</Typography>
        )}
      </Box>
      <LinearProgress
        variant="determinate"
        value={clampedValue}
        color={autoColor}
        sx={{ height, borderRadius: height / 2, backgroundColor: '#E0E0E0' }}
      />
    </Box>
  );
};

export default ProgressBar;
