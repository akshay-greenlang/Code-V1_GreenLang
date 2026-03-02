/**
 * WaterfallChart - Scope 3 category waterfall visualization
 *
 * Renders a Recharts BarChart with 15 bars representing each Scope 3
 * category sorted by emissions magnitude. Includes a cumulative line
 * overlay, color gradient from largest to smallest, and an optional
 * materiality threshold reference line.
 */

import React, { useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import { Box, Typography } from '@mui/material';
import { SCOPE3_CATEGORY_NAMES } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface WaterfallChartProps {
  categories: Record<string, number>;
  materialityThreshold?: number;
  height?: number;
}

interface WaterfallEntry {
  key: string;
  label: string;
  shortLabel: string;
  value: number;
  cumulative: number;
  color: string;
}

const COLOR_SCALE = [
  '#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#4caf50',
  '#66bb6a', '#81c784', '#a5d6a7', '#aed581', '#c5e1a5',
  '#dce775', '#e6ee9c', '#f0f4c3', '#fff9c4', '#fff59d',
];

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: WaterfallEntry }>;
}) => {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        p: 1.5,
        borderRadius: 1,
        boxShadow: 2,
        border: '1px solid',
        borderColor: 'divider',
        maxWidth: 260,
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        {item.label}
      </Typography>
      <Typography variant="body2">
        {formatNumber(item.value)} tCO2e
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Cumulative: {formatNumber(item.cumulative)} tCO2e
      </Typography>
    </Box>
  );
};

const WaterfallChart: React.FC<WaterfallChartProps> = ({
  categories,
  materialityThreshold,
  height = 360,
}) => {
  const data: WaterfallEntry[] = useMemo(() => {
    const sorted = Object.entries(categories)
      .map(([key, value]) => ({ key, value }))
      .sort((a, b) => b.value - a.value);

    let cumulative = 0;
    return sorted.map((entry, index) => {
      cumulative += entry.value;
      const catNum = entry.key.replace('cat_', '');
      const fullName = SCOPE3_CATEGORY_NAMES[entry.key] || entry.key;
      return {
        key: entry.key,
        label: `Cat ${catNum}: ${fullName}`,
        shortLabel: `Cat ${catNum}`,
        value: entry.value,
        cumulative,
        color: COLOR_SCALE[Math.min(index, COLOR_SCALE.length - 1)],
      };
    });
  }, [categories]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart
        data={data}
        margin={{ top: 10, right: 30, left: 10, bottom: 60 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis
          dataKey="shortLabel"
          tick={{ fontSize: 11, angle: -45, textAnchor: 'end' }}
          interval={0}
          height={60}
        />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 12 }}
          tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
          label={{
            value: 'tCO2e',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 12 },
          }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 12 }}
          tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
          label={{
            value: 'Cumulative tCO2e',
            angle: 90,
            position: 'insideRight',
            style: { fontSize: 12 },
          }}
        />
        <Tooltip content={<CustomTooltip />} />

        <Bar dataKey="value" yAxisId="left" radius={[4, 4, 0, 0]} barSize={28}>
          {data.map((entry, index) => (
            <Cell key={`bar-${index}`} fill={entry.color} />
          ))}
        </Bar>

        <Line
          type="monotone"
          dataKey="cumulative"
          yAxisId="right"
          stroke="#ef6c00"
          strokeWidth={2}
          dot={{ fill: '#ef6c00', r: 3 }}
          name="Cumulative"
        />

        {materialityThreshold !== undefined && (
          <ReferenceLine
            y={materialityThreshold}
            yAxisId="left"
            stroke="#c62828"
            strokeDasharray="6 3"
            strokeWidth={1.5}
            label={{
              value: `Materiality: ${formatNumber(materialityThreshold)} tCO2e`,
              position: 'top',
              style: { fontSize: 11, fill: '#c62828' },
            }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default WaterfallChart;
