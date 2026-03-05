/**
 * TemperatureGauge - Temperature alignment dial 0-4C for dashboard.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { getTemperatureColor } from '../../utils/pathwayHelpers';
import { formatPathwayAlignment } from '../../utils/formatters';
import type { PathwayAlignment } from '../../types';

interface TemperatureGaugeProps {
  temperature: number;
  alignment: PathwayAlignment;
  trend?: { year: number; score: number }[];
}

const TemperatureGauge: React.FC<TemperatureGaugeProps> = ({ temperature, alignment }) => {
  const color = getTemperatureColor(temperature);
  const pct = Math.min((temperature / 4.0) * 100, 100);
  const size = 140;
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashoffset = circumference - (pct / 100) * circumference;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Temperature Score</Typography>
        <Box sx={{ position: 'relative', width: size, height: size }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#E0E0E0" strokeWidth="10" />
            <circle
              cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color} strokeWidth="10"
              strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={dashoffset}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
              style={{ transition: 'stroke-dashoffset 0.6s ease-in-out' }}
            />
          </svg>
          <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
            <Typography variant="h4" sx={{ fontWeight: 700, color }}>{temperature.toFixed(2)}{'\u00B0C'}</Typography>
          </Box>
        </Box>
        <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 600, color }}>
          {formatPathwayAlignment(alignment)}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Implied Temperature Rise
        </Typography>
      </CardContent>
    </Card>
  );
};

export default TemperatureGauge;
