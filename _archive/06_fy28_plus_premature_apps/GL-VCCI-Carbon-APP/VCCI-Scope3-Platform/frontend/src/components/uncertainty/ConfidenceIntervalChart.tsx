/**
 * ConfidenceIntervalChart - Emissions trend with shaded confidence bands
 *
 * Displays time-series emissions data with layered confidence interval
 * regions. Supports toggling between 90%, 95%, and 99% CI levels with
 * graduated shading. The central line shows the mean or median value,
 * and category-level or aggregate views are supported through props.
 */

import React, { useState, useMemo } from 'react';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  Paper,
  Typography,
  Box,
  ToggleButtonGroup,
  ToggleButton,
  Chip,
  Stack,
} from '@mui/material';

// ==============================================================================
// Types
// ==============================================================================

interface ConfidenceIntervalDataPoint {
  period: string;
  mean: number;
  p5: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
}

interface ConfidenceIntervalChartProps {
  data: ConfidenceIntervalDataPoint[];
  unit?: string;
  title?: string;
}

type CILevel = 90 | 95 | 99;

// ==============================================================================
// Constants
// ==============================================================================

const CI_CONFIG: Record<CILevel, { lower: keyof ConfidenceIntervalDataPoint; upper: keyof ConfidenceIntervalDataPoint; color: string; label: string }> = {
  90: { lower: 'p5', upper: 'p95', color: '#1976d2', label: '90% CI (P5-P95)' },
  95: { lower: 'p25', upper: 'p75', color: '#1565c0', label: '50% CI (P25-P75)' },
  99: { lower: 'p10', upper: 'p90', color: '#0d47a1', label: '80% CI (P10-P90)' },
};

// ==============================================================================
// Component
// ==============================================================================

const ConfidenceIntervalChart: React.FC<ConfidenceIntervalChartProps> = ({
  data,
  unit = 'tCO2e',
  title = 'Emissions Trend with Confidence Intervals',
}) => {
  const [selectedCILevels, setSelectedCILevels] = useState<CILevel[]>([90, 95]);

  // Prepare chart data with range fields for area rendering
  const chartData = useMemo(() => {
    return data.map((point) => ({
      ...point,
      // 90% CI range (widest)
      ci90Range: [point.p5, point.p95],
      // 80% CI range (medium)
      ci80Range: [point.p10, point.p90],
      // 50% CI range (narrowest)
      ci50Range: [point.p25, point.p75],
    }));
  }, [data]);

  // Summary stats
  const summaryStats = useMemo(() => {
    if (data.length === 0) return null;
    const latest = data[data.length - 1];
    const earliest = data[0];
    const trend = ((latest.mean - earliest.mean) / earliest.mean) * 100;
    const avgSpread = data.reduce((sum, d) => sum + (d.p95 - d.p5), 0) / data.length;

    return {
      latestMean: latest.mean,
      trend,
      avgSpread,
      periods: data.length,
    };
  }, [data]);

  const handleCIChange = (_: React.MouseEvent<HTMLElement>, newLevels: CILevel[]) => {
    if (newLevels.length > 0) {
      setSelectedCILevels(newLevels);
    }
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    const point = payload[0]?.payload as ConfidenceIntervalDataPoint;
    if (!point) return null;

    return (
      <Paper sx={{ p: 1.5 }} elevation={3}>
        <Typography variant="body2" fontWeight="bold" gutterBottom>
          {label}
        </Typography>
        <Typography variant="body2" color="primary">
          Mean: {point.mean.toFixed(2)} {unit}
        </Typography>
        <Typography variant="body2" color="secondary">
          Median (P50): {point.p50.toFixed(2)} {unit}
        </Typography>
        <Box sx={{ mt: 0.5 }}>
          <Typography variant="caption" color="textSecondary">
            P5: {point.p5.toFixed(2)} | P10: {point.p10.toFixed(2)} | P25: {point.p25.toFixed(2)}
          </Typography>
        </Box>
        <Box>
          <Typography variant="caption" color="textSecondary">
            P75: {point.p75.toFixed(2)} | P90: {point.p90.toFixed(2)} | P95: {point.p95.toFixed(2)}
          </Typography>
        </Box>
        <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, display: 'block' }}>
          90% CI width: {(point.p95 - point.p5).toFixed(2)} {unit}
        </Typography>
      </Paper>
    );
  };

  if (data.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="textSecondary">
          No confidence interval data available.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          {summaryStats && (
            <Stack direction="row" spacing={1}>
              <Chip
                label={`Latest: ${summaryStats.latestMean.toFixed(2)} ${unit}`}
                size="small"
                color="primary"
                variant="outlined"
              />
              <Chip
                label={`Trend: ${summaryStats.trend >= 0 ? '+' : ''}${summaryStats.trend.toFixed(1)}%`}
                size="small"
                color={summaryStats.trend > 0 ? 'error' : 'success'}
                variant="outlined"
              />
              <Chip
                label={`Avg CI width: ${summaryStats.avgSpread.toFixed(2)} ${unit}`}
                size="small"
                variant="outlined"
              />
            </Stack>
          )}
        </Box>

        {/* CI Level Toggle */}
        <ToggleButtonGroup
          value={selectedCILevels}
          onChange={handleCIChange}
          size="small"
          aria-label="Confidence interval levels"
        >
          <ToggleButton value={90} aria-label="90% CI">
            90% CI
          </ToggleButton>
          <ToggleButton value={99} aria-label="80% CI">
            80% CI
          </ToggleButton>
          <ToggleButton value={95} aria-label="50% CI">
            50% CI
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
        >
          <defs>
            <linearGradient id="ci90Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1976d2" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#1976d2" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="ci80Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1565c0" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#1565c0" stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="ci50Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0d47a1" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#0d47a1" stopOpacity={0.15} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="period"
            tick={{ fontSize: 12 }}
            label={{ value: 'Period', position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            label={{ value: unit, angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* 90% CI band (widest, lightest) */}
          {selectedCILevels.includes(90) && (
            <>
              <Area
                type="monotone"
                dataKey="p95"
                stroke="none"
                fill="url(#ci90Gradient)"
                name="90% CI Upper (P95)"
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="p5"
                stroke="none"
                fill="#ffffff"
                name="90% CI Lower (P5)"
                legendType="none"
              />
            </>
          )}

          {/* 80% CI band (medium) */}
          {selectedCILevels.includes(99) && (
            <>
              <Area
                type="monotone"
                dataKey="p90"
                stroke="none"
                fill="url(#ci80Gradient)"
                name="80% CI Upper (P90)"
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="p10"
                stroke="none"
                fill="#ffffff"
                name="80% CI Lower (P10)"
                legendType="none"
              />
            </>
          )}

          {/* 50% CI band (narrowest, darkest) */}
          {selectedCILevels.includes(95) && (
            <>
              <Area
                type="monotone"
                dataKey="p75"
                stroke="none"
                fill="url(#ci50Gradient)"
                name="50% CI Upper (P75)"
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="p25"
                stroke="none"
                fill="#ffffff"
                name="50% CI Lower (P25)"
                legendType="none"
              />
            </>
          )}

          {/* Central line: Mean */}
          <Line
            type="monotone"
            dataKey="mean"
            stroke="#1976d2"
            strokeWidth={2.5}
            dot={{ fill: '#1976d2', r: 3 }}
            activeDot={{ r: 6 }}
            name={`Mean (${unit})`}
          />

          {/* Median line */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke="#9c27b0"
            strokeWidth={1.5}
            strokeDasharray="5 5"
            dot={false}
            name={`Median (${unit})`}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* CI Level Legend */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, mt: 1 }}>
        {selectedCILevels.includes(90) && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 20, height: 12, backgroundColor: '#1976d2', opacity: 0.15, borderRadius: 0.5 }} />
            <Typography variant="caption">90% CI (P5-P95)</Typography>
          </Stack>
        )}
        {selectedCILevels.includes(99) && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 20, height: 12, backgroundColor: '#1565c0', opacity: 0.25, borderRadius: 0.5 }} />
            <Typography variant="caption">80% CI (P10-P90)</Typography>
          </Stack>
        )}
        {selectedCILevels.includes(95) && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 20, height: 12, backgroundColor: '#0d47a1', opacity: 0.35, borderRadius: 0.5 }} />
            <Typography variant="caption">50% CI (P25-P75)</Typography>
          </Stack>
        )}
      </Box>
    </Paper>
  );
};

export default ConfidenceIntervalChart;
