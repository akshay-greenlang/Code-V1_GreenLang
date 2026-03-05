/**
 * ThresholdEvaluator - Shows actual vs required threshold values with visual bars.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress, Chip } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';

const DEMO_THRESHOLDS = [
  { criterion: 'GHG emissions intensity', actual: 15, required: 100, unit: 'gCO2e/kWh', operator: 'lte' as const, passes: true },
  { criterion: 'Energy efficiency', actual: 92, required: 80, unit: '%', operator: 'gte' as const, passes: true },
  { criterion: 'Water consumption', actual: 1.2, required: 2.0, unit: 'm3/MWh', operator: 'lte' as const, passes: true },
  { criterion: 'Waste recycling rate', actual: 65, required: 70, unit: '%', operator: 'gte' as const, passes: false },
];

interface ThresholdEvaluatorProps {
  thresholds?: typeof DEMO_THRESHOLDS;
}

const ThresholdEvaluator: React.FC<ThresholdEvaluatorProps> = ({ thresholds = DEMO_THRESHOLDS }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Threshold Evaluation
      </Typography>
      {thresholds.map((t, idx) => {
        const pct = t.operator === 'lte'
          ? Math.min((1 - t.actual / t.required) * 100, 100)
          : Math.min((t.actual / t.required) * 100, 100);
        const margin = t.operator === 'lte'
          ? ((t.required - t.actual) / t.required * 100).toFixed(1)
          : ((t.actual - t.required) / t.required * 100).toFixed(1);

        return (
          <Box key={idx} sx={{ mb: 2.5 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {t.passes ? <CheckCircle fontSize="small" sx={{ color: '#2E7D32' }} /> : <Cancel fontSize="small" sx={{ color: '#C62828' }} />}
                <Typography variant="body2" sx={{ fontWeight: 500 }}>{t.criterion}</Typography>
              </Box>
              <Chip
                label={`Margin: ${t.passes ? '+' : ''}${margin}%`}
                size="small"
                color={t.passes ? 'success' : 'error'}
                variant="outlined"
              />
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{ flexGrow: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={Math.abs(pct)}
                  color={t.passes ? 'success' : 'error'}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
              <Typography variant="caption" sx={{ minWidth: 140, textAlign: 'right' }}>
                {t.actual} / {t.required} {t.unit}
              </Typography>
            </Box>
          </Box>
        );
      })}
    </CardContent>
  </Card>
);

export default ThresholdEvaluator;
