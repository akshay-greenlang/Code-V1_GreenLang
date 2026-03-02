/**
 * SourceBreakdown - Horizontal bar chart of Scope 1 source categories
 *
 * Renders a Recharts BarChart with horizontal bars for each Scope 1
 * source category (stationary combustion, mobile combustion, process
 * emissions, fugitive emissions), sorted by magnitude. Each bar shows
 * the percentage of total Scope 1 emissions.
 */

import React, { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Box, Typography, Card, CardContent } from '@mui/material';
import type { Scope1CategoryBreakdown } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface SourceBreakdownProps {
  categories: Scope1CategoryBreakdown[];
  onCategoryClick?: (category: string) => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  stationary_combustion: 'Stationary Combustion',
  mobile_combustion: 'Mobile Combustion',
  process_emissions: 'Process Emissions',
  fugitive_emissions: 'Fugitive Emissions',
};

const CATEGORY_COLORS: Record<string, string> = {
  stationary_combustion: '#e53935',
  mobile_combustion: '#ef6c00',
  process_emissions: '#8e24aa',
  fugitive_emissions: '#1e88e5',
};

interface ChartEntry {
  key: string;
  label: string;
  value: number;
  percentage: number;
  color: string;
}

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartEntry }>;
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
      }}
    >
      <Typography variant="subtitle2">{item.label}</Typography>
      <Typography variant="body2">
        {formatNumber(item.value)} tCO2e ({item.percentage.toFixed(1)}%)
      </Typography>
    </Box>
  );
};

const SourceBreakdown: React.FC<SourceBreakdownProps> = ({ categories, onCategoryClick }) => {
  const data: ChartEntry[] = useMemo(() => {
    return [...categories]
      .sort((a, b) => b.emissions_tco2e - a.emissions_tco2e)
      .map((cat) => ({
        key: cat.category,
        label: CATEGORY_LABELS[cat.category] || cat.category,
        value: cat.emissions_tco2e,
        percentage: cat.percentage_of_total,
        color: CATEGORY_COLORS[cat.category] || '#757575',
      }));
  }, [categories]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Scope 1 Source Categories
        </Typography>
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 60 + 40)}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 60, left: 140, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 12 }}
              tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
            />
            <YAxis
              dataKey="label"
              type="category"
              tick={{ fontSize: 12 }}
              width={130}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="value"
              radius={[0, 4, 4, 0]}
              barSize={24}
              onClick={(entry: ChartEntry) => onCategoryClick?.(entry.key)}
              cursor={onCategoryClick ? 'pointer' : 'default'}
              label={{
                position: 'right',
                formatter: (v: number) => `${((v / (data[0]?.value || 1)) * data[0]?.percentage || 0).toFixed(0)}%`,
                fontSize: 11,
                fill: '#4a4a68',
              }}
            >
              {data.map((entry) => (
                <Cell key={entry.key} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default SourceBreakdown;
