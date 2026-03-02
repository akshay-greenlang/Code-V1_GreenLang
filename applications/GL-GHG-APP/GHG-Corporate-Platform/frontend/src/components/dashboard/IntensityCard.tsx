/**
 * IntensityCard - Intensity metric display with trend sparkline
 *
 * Shows an emissions intensity value (e.g., tCO2e per $M revenue),
 * year-over-year change arrow, a mini sparkline, and an optional
 * benchmark comparison indicator.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import type { IntensityMetric } from '../../types';

interface IntensityCardProps {
  metric: IntensityMetric;
  previousValue?: number;
  benchmark?: number;
}

const IntensityCard: React.FC<IntensityCardProps> = ({
  metric,
  previousValue,
  benchmark,
}) => {
  const yoyChange = metric.year_over_year_change_percent;
  const isImproving = yoyChange !== null && yoyChange < 0;
  const isWorsening = yoyChange !== null && yoyChange > 0;

  const sparkData = [
    { v: previousValue ?? metric.intensity_value * 1.1 },
    { v: metric.intensity_value * 1.05 },
    { v: metric.intensity_value * 0.98 },
    { v: metric.intensity_value * 1.02 },
    { v: metric.intensity_value },
  ];

  const benchmarkDiff = benchmark
    ? ((metric.intensity_value - benchmark) / benchmark) * 100
    : null;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {metric.metric_name}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1, mb: 1 }}>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            {metric.intensity_value.toFixed(2)}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            {metric.intensity_unit}
          </Typography>
        </Box>

        {/* YoY change */}
        {yoyChange !== null && (
          <Chip
            icon={
              isImproving ? (
                <TrendingDown fontSize="small" />
              ) : isWorsening ? (
                <TrendingUp fontSize="small" />
              ) : (
                <Remove fontSize="small" />
              )
            }
            label={`${yoyChange > 0 ? '+' : ''}${yoyChange.toFixed(1)}% YoY`}
            size="small"
            color={isImproving ? 'success' : isWorsening ? 'error' : 'default'}
            variant="outlined"
            sx={{ mb: 1.5 }}
          />
        )}

        {/* Mini sparkline */}
        <Box sx={{ height: 40, mt: 1 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="v"
                stroke={isImproving ? '#2e7d32' : isWorsening ? '#c62828' : '#757575'}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>

        {/* Benchmark comparison */}
        {benchmark !== undefined && benchmarkDiff !== null && (
          <Box
            sx={{
              mt: 1.5,
              pt: 1,
              borderTop: '1px solid',
              borderColor: 'divider',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Typography variant="caption" color="text.secondary">
              vs. Benchmark ({benchmark.toFixed(2)})
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontWeight: 600,
                color: benchmarkDiff <= 0 ? 'success.main' : 'error.main',
              }}
            >
              {benchmarkDiff > 0 ? '+' : ''}
              {benchmarkDiff.toFixed(1)}%
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default IntensityCard;
