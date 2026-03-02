/**
 * TrendChart - Multi-year emissions trend visualization
 *
 * Renders a Recharts AreaChart with three stacked areas for
 * Scope 1/2/3 emissions over time. Supports an optional dashed
 * target line and a tooltip showing per-year scope breakdown.
 */

import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Box, Typography } from '@mui/material';
import type { TrendDataPoint } from '../../types';
import { SCOPE_COLORS } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface TrendChartProps {
  data: TrendDataPoint[];
  targetLine?: number;
  height?: number;
}

interface ChartDataPoint {
  label: string;
  scope1: number;
  scope2: number;
  scope3: number;
  total: number;
}

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;

  const total = payload.reduce((sum, p) => sum + p.value, 0);

  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        p: 1.5,
        borderRadius: 1,
        boxShadow: 2,
        border: '1px solid',
        borderColor: 'divider',
        minWidth: 180,
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        {label}
      </Typography>
      {payload.map((entry) => (
        <Box
          key={entry.name}
          sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}
        >
          <Typography variant="body2" sx={{ color: entry.color }}>
            {entry.name}
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {formatNumber(entry.value)}
          </Typography>
        </Box>
      ))}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: 2,
          mt: 0.5,
          pt: 0.5,
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          Total
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 700 }}>
          {formatNumber(total)} tCO2e
        </Typography>
      </Box>
    </Box>
  );
};

const TrendChart: React.FC<TrendChartProps> = ({ data, targetLine, height = 320 }) => {
  const chartData: ChartDataPoint[] = data.map((d) => ({
    label: d.period_label || String(d.year),
    scope1: d.scope1_tco2e,
    scope2: d.scope2_location_tco2e,
    scope3: d.scope3_tco2e,
    total: d.total_tco2e,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="gradScope1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={SCOPE_COLORS.scope1} stopOpacity={0.3} />
            <stop offset="95%" stopColor={SCOPE_COLORS.scope1} stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="gradScope2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={SCOPE_COLORS.scope2} stopOpacity={0.3} />
            <stop offset="95%" stopColor={SCOPE_COLORS.scope2} stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="gradScope3" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={SCOPE_COLORS.scope3} stopOpacity={0.3} />
            <stop offset="95%" stopColor={SCOPE_COLORS.scope3} stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey="label" tick={{ fontSize: 12 }} />
        <YAxis
          tick={{ fontSize: 12 }}
          tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
          label={{ value: 'tCO2e', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          formatter={(value: string) => (
            <span style={{ fontSize: 12 }}>{value}</span>
          )}
        />
        <Area
          type="monotone"
          dataKey="scope1"
          name="Scope 1"
          stackId="1"
          stroke={SCOPE_COLORS.scope1}
          fill="url(#gradScope1)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="scope2"
          name="Scope 2"
          stackId="1"
          stroke={SCOPE_COLORS.scope2}
          fill="url(#gradScope2)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="scope3"
          name="Scope 3"
          stackId="1"
          stroke={SCOPE_COLORS.scope3}
          fill="url(#gradScope3)"
          strokeWidth={2}
        />
        {targetLine !== undefined && (
          <ReferenceLine
            y={targetLine}
            stroke="#ff9800"
            strokeDasharray="8 4"
            strokeWidth={2}
            label={{
              value: `Target: ${formatNumber(targetLine)} tCO2e`,
              position: 'right',
              style: { fontSize: 11, fill: '#ff9800' },
            }}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default TrendChart;
