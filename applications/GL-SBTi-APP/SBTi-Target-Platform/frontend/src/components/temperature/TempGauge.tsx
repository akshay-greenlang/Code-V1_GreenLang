/**
 * TempGauge - Temperature dial 0-4C with color bands.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { getTemperatureColor } from '../../utils/pathwayHelpers';

interface TempGaugeProps { temperature: number; label?: string; }

const TempGauge: React.FC<TempGaugeProps> = ({ temperature, label = 'Temperature Score' }) => {
  const color = getTemperatureColor(temperature);
  const pct = Math.min((temperature / 4.0) * 100, 100);
  const bands = [
    { max: 1.5, color: '#1B5E20', label: '1.5C' }, { max: 2.0, color: '#2E7D32', label: '2C' },
    { max: 2.5, color: '#EF6C00', label: '2.5C' }, { max: 3.0, color: '#E65100', label: '3C' },
    { max: 4.0, color: '#B71C1C', label: '4C' },
  ];
  return (
    <Card>
      <CardContent sx={{ textAlign: 'center' }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>{label}</Typography>
        <Typography variant="h2" sx={{ fontWeight: 700, color }}>{temperature.toFixed(2)}{'\u00B0C'}</Typography>
        <Box sx={{ display: 'flex', mt: 2, borderRadius: 1, overflow: 'hidden', height: 12 }}>
          {bands.map((b, i) => (
            <Box key={i} sx={{ flex: 1, backgroundColor: b.color, position: 'relative' }}>
              {temperature <= b.max && temperature > (i > 0 ? bands[i - 1].max : 0) && (
                <Box sx={{ position: 'absolute', top: -4, left: '50%', transform: 'translateX(-50%)', width: 4, height: 20, backgroundColor: 'white', borderRadius: 2 }} />
              )}
            </Box>
          ))}
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
          {bands.map((b) => <Typography key={b.label} variant="caption" color="text.secondary">{b.label}</Typography>)}
        </Box>
      </CardContent>
    </Card>
  );
};

export default TempGauge;
