/**
 * TargetProgress - Target tracking dashboard
 *
 * Shows active targets with progress bars, a forecast trajectory
 * line chart comparing actual vs. required path, gap-to-target
 * alert, and annual reduction rate display.
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Chip,
  Alert,
} from '@mui/material';
import { CheckCircle, Warning, TrendingDown } from '@mui/icons-material';
import type { Target, TargetProgress as TargetProgressType, ForecastPoint } from '../../types';
import { formatNumber, formatPercent } from '../../utils/formatters';

interface TargetProgressProps {
  targets: Target[];
  progress: Record<string, TargetProgressType>;
}

const TargetProgressComponent: React.FC<TargetProgressProps> = ({ targets, progress }) => {
  const activeTargets = targets.filter((t) => t.status !== 'achieved');

  return (
    <Box>
      {/* Active targets with progress bars */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {activeTargets.map((target) => {
          const tp = progress[target.id];
          const progressPercent = tp?.progress_percent ?? target.progress_percent;
          const onTrack = tp?.on_track ?? target.status === 'on_track';
          const gap = tp?.gap_to_target ?? 0;

          return (
            <Grid item xs={12} md={6} key={target.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {target.name}
                    </Typography>
                    <Chip
                      icon={onTrack ? <CheckCircle fontSize="small" /> : <Warning fontSize="small" />}
                      label={onTrack ? 'On Track' : 'Behind'}
                      size="small"
                      color={onTrack ? 'success' : 'error'}
                      variant="outlined"
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" color="text.secondary">
                        Progress: {formatPercent(progressPercent)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Target: -{target.target_reduction_percent.toFixed(0)}% by {target.target_year}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(progressPercent, 100)}
                      sx={{
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 5,
                          backgroundColor: onTrack ? '#2e7d32' : '#c62828',
                        },
                      }}
                    />
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Typography variant="caption" color="text.secondary">Required</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {target.annual_reduction_rate.toFixed(1)}%/yr
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="caption" color="text.secondary">Actual</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {tp ? tp.actual_annual_reduction.toFixed(1) : '-'}%/yr
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="caption" color="text.secondary">Gap</Typography>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 600,
                          color: gap > 0 ? 'error.main' : 'success.main',
                        }}
                      >
                        {gap > 0 ? '+' : ''}{formatNumber(gap)} tCO2e
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Forecast trajectory chart for first target with data */}
      {activeTargets.length > 0 && (() => {
        const firstTarget = activeTargets[0];
        const tp = progress[firstTarget.id];
        const forecastData = tp?.forecast_data || firstTarget.interim_targets?.map((it) => ({
          year: it.year,
          required: it.target_emissions_tco2e,
          actual: it.actual_emissions_tco2e ?? undefined,
        })) || [];

        if (forecastData.length === 0) return null;

        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Forecast Trajectory: {firstTarget.name}
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={forecastData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
                    label={{ value: 'tCO2e', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [
                      `${formatNumber(value)} tCO2e`,
                      name === 'required' ? 'Required Path' : name === 'actual' ? 'Actual' : 'Forecast',
                    ]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="required"
                    name="Required Path"
                    stroke="#ef6c00"
                    strokeDasharray="8 4"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Actual"
                    stroke="#1b5e20"
                    strokeWidth={2.5}
                    dot={{ fill: '#1b5e20', r: 4 }}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast"
                    stroke="#1e88e5"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        );
      })()}

      {activeTargets.length === 0 && (
        <Alert severity="info">
          No active targets. Set a reduction target to begin tracking progress.
        </Alert>
      )}
    </Box>
  );
};

export default TargetProgressComponent;
