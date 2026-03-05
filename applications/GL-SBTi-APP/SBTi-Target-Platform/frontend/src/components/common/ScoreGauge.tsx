/**
 * ScoreGauge - Circular gauge for readiness/temperature scores.
 */

import React from 'react';
import { Box, Typography } from '@mui/material';

interface ScoreGaugeProps {
  value: number;
  maxValue?: number;
  label: string;
  subtitle?: string;
  size?: number;
  color?: string;
  unit?: string;
}

const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  value, maxValue = 100, label, subtitle, size = 160, color, unit = '%',
}) => {
  const pct = Math.min((value / maxValue) * 100, 100);
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashoffset = circumference - (pct / 100) * circumference;

  const autoColor = color || (pct >= 75 ? '#2E7D32' : pct >= 50 ? '#EF6C00' : '#C62828');

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke="#E0E0E0" strokeWidth="8"
          />
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke={autoColor} strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashoffset}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            style={{ transition: 'stroke-dashoffset 0.6s ease-in-out' }}
          />
        </svg>
        <Box sx={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)', textAlign: 'center',
        }}>
          <Typography variant="h5" sx={{ fontWeight: 700, color: autoColor }}>
            {typeof value === 'number' ? value.toFixed(unit === '\u00B0C' ? 2 : 0) : value}{unit}
          </Typography>
        </Box>
      </Box>
      <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 600 }}>{label}</Typography>
      {subtitle && <Typography variant="caption" color="text.secondary">{subtitle}</Typography>}
    </Box>
  );
};

export default ScoreGauge;
