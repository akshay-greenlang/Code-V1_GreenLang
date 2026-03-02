/**
 * GapAnalysis - Gap-to-target analysis visualization
 *
 * Displays a bar chart comparing current emissions vs. target emissions
 * per scope, a waterfall breakdown of the gap, required vs. actual
 * annual reduction rates, and scenario projections showing when the
 * target will be met at current vs. required reduction pace.
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Alert,
  LinearProgress,
  Chip,
  Divider,
} from '@mui/material';
import {
  TrendingDown,
  TrendingUp,
  Speed,
  Flag,
  Timeline,
} from '@mui/icons-material';
import type { Target, TargetProgress } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface GapAnalysisProps {
  target: Target;
  progress: TargetProgress;
}

interface GapWaterfallItem {
  name: string;
  value: number;
  color: string;
  type: 'base' | 'reduction' | 'gap' | 'target';
}

const GapAnalysis: React.FC<GapAnalysisProps> = ({ target, progress }) => {
  const gap = progress.gap_to_target;
  const onTrack = progress.on_track;
  const yearsRemaining = target.target_year - new Date().getFullYear();
  const totalReductionNeeded = target.base_year_emissions_tco2e - target.target_emissions_tco2e;
  const reductionAchieved = target.base_year_emissions_tco2e - progress.current_emissions_tco2e;
  const reductionPercent = totalReductionNeeded > 0
    ? (reductionAchieved / totalReductionNeeded) * 100
    : 0;

  // Waterfall data: Base Year -> Achieved Reduction -> Remaining Gap -> Target
  const waterfallData = useMemo<GapWaterfallItem[]>(() => [
    {
      name: `Base Year (${target.base_year})`,
      value: target.base_year_emissions_tco2e,
      color: '#757575',
      type: 'base',
    },
    {
      name: 'Reduction Achieved',
      value: Math.max(reductionAchieved, 0),
      color: '#2e7d32',
      type: 'reduction',
    },
    {
      name: 'Remaining Gap',
      value: Math.max(gap, 0),
      color: onTrack ? '#1e88e5' : '#c62828',
      type: 'gap',
    },
    {
      name: `Target (${target.target_year})`,
      value: target.target_emissions_tco2e,
      color: '#ef6c00',
      type: 'target',
    },
  ], [target, progress, gap, onTrack, reductionAchieved]);

  // Comparison bar chart data: current vs target per metric
  const comparisonData = useMemo(() => [
    {
      metric: 'Total Emissions',
      current: progress.current_emissions_tco2e,
      target: target.target_emissions_tco2e,
    },
    {
      metric: 'Annual Reduction',
      current: progress.actual_annual_reduction,
      target: progress.required_annual_reduction,
    },
  ], [progress, target]);

  // Projected year of achievement at current rate
  const projectedYear = useMemo(() => {
    if (progress.actual_annual_reduction <= 0) return null;
    const remainingReductionNeeded = gap;
    if (remainingReductionNeeded <= 0) return new Date().getFullYear();
    const yearsNeeded = remainingReductionNeeded /
      (progress.current_emissions_tco2e * (progress.actual_annual_reduction / 100));
    return Math.ceil(new Date().getFullYear() + yearsNeeded);
  }, [progress, gap]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Gap-to-Target Analysis</Typography>
          <Chip
            icon={onTrack ? <TrendingDown fontSize="small" /> : <TrendingUp fontSize="small" />}
            label={onTrack ? 'On Track' : 'Action Required'}
            size="small"
            color={onTrack ? 'success' : 'error'}
          />
        </Box>

        {/* Status alert */}
        <Alert
          severity={onTrack ? 'success' : 'warning'}
          icon={onTrack ? <TrendingDown /> : <Speed />}
          sx={{ mb: 3 }}
        >
          {onTrack
            ? `Emissions reductions are on track. Current trajectory meets the ${target.target_reduction_percent.toFixed(0)}% reduction target by ${target.target_year}.`
            : `A gap of ${formatNumber(Math.abs(gap))} tCO2e remains to reach the ${target.target_reduction_percent.toFixed(0)}% reduction target by ${target.target_year}. Annual reduction rate needs to increase from ${progress.actual_annual_reduction.toFixed(1)}% to ${progress.required_annual_reduction.toFixed(1)}%.`
          }
        </Alert>

        {/* KPI cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="caption" color="text.secondary">Gap to Target</Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 700,
                  color: gap > 0 ? 'error.main' : 'success.main',
                }}
              >
                {gap > 0 ? '+' : ''}{formatNumber(gap)}
              </Typography>
              <Typography variant="caption" color="text.secondary">tCO2e</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="caption" color="text.secondary">Reduction Achieved</Typography>
              <Typography variant="h6" sx={{ fontWeight: 700, color: 'success.main' }}>
                {reductionPercent.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                of {target.target_reduction_percent.toFixed(0)}% target
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="caption" color="text.secondary">Years Remaining</Typography>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                {yearsRemaining}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                to {target.target_year}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="caption" color="text.secondary">Projected Achievement</Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 700,
                  color: projectedYear && projectedYear <= target.target_year
                    ? 'success.main'
                    : 'error.main',
                }}
              >
                {projectedYear ?? 'N/A'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {projectedYear && projectedYear <= target.target_year ? 'Before deadline' : 'After deadline'}
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Progress bar */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="body2" color="text.secondary">
              Reduction Progress
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {reductionPercent.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={Math.min(Math.max(reductionPercent, 0), 100)}
            sx={{
              height: 12,
              borderRadius: 6,
              backgroundColor: '#e0e0e0',
              '& .MuiLinearProgress-bar': {
                borderRadius: 6,
                backgroundColor: onTrack ? '#2e7d32' : '#ef6c00',
              },
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              {formatNumber(target.base_year_emissions_tco2e)} tCO2e ({target.base_year})
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatNumber(target.target_emissions_tco2e)} tCO2e ({target.target_year})
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Waterfall chart: base -> reduction -> gap -> target */}
        <Typography variant="subtitle2" gutterBottom>
          Emissions Waterfall
        </Typography>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={waterfallData}
            margin={{ top: 10, right: 20, left: 10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11 }}
              interval={0}
              angle={0}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v)
              }
              label={{
                value: 'tCO2e',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 11 },
              }}
            />
            <Tooltip
              formatter={(value: number) => [
                `${formatNumber(value)} tCO2e`,
                '',
              ]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {waterfallData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <Divider sx={{ my: 3 }} />

        {/* Reduction rate comparison */}
        <Typography variant="subtitle2" gutterBottom>
          Annual Reduction Rate
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
                textAlign: 'center',
              }}
            >
              <Flag color="warning" sx={{ mb: 0.5 }} />
              <Typography variant="caption" color="text.secondary" display="block">
                Required Rate
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'warning.main' }}>
                {progress.required_annual_reduction.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">per year</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: '1px solid',
                borderColor: progress.actual_annual_reduction >= progress.required_annual_reduction
                  ? 'success.main'
                  : 'error.main',
                textAlign: 'center',
              }}
            >
              <Timeline
                color={
                  progress.actual_annual_reduction >= progress.required_annual_reduction
                    ? 'success'
                    : 'error'
                }
                sx={{ mb: 0.5 }}
              />
              <Typography variant="caption" color="text.secondary" display="block">
                Actual Rate
              </Typography>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 700,
                  color: progress.actual_annual_reduction >= progress.required_annual_reduction
                    ? 'success.main'
                    : 'error.main',
                }}
              >
                {progress.actual_annual_reduction.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">per year</Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default GapAnalysis;
