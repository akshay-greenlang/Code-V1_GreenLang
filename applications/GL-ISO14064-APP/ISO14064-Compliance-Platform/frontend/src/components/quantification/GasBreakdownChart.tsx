/**
 * GasBreakdownChart - Recharts pie/donut chart for gas breakdown
 *
 * Renders a donut chart showing the distribution of emissions by
 * GHG gas type (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3) using the
 * GAS_COLORS palette from types.
 */

import React, { useMemo } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box, Typography } from '@mui/material';
import { GHGGas, GAS_COLORS } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface GasBreakdownChartProps {
  data: Record<string, number>;
  height?: number;
  innerRadius?: number;
  outerRadius?: number;
  title?: string;
}

const GasBreakdownChart: React.FC<GasBreakdownChartProps> = ({
  data,
  height = 300,
  innerRadius = 55,
  outerRadius = 110,
  title,
}) => {
  const chartData = useMemo(() => {
    return Object.entries(data)
      .filter(([, v]) => v > 0)
      .map(([gas, value]) => ({
        name: gas,
        value,
        color: GAS_COLORS[gas as GHGGas] || '#9e9e9e',
      }))
      .sort((a, b) => b.value - a.value);
  }, [data]);

  const total = chartData.reduce((sum, d) => sum + d.value, 0);

  if (chartData.length === 0) {
    return (
      <Box sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No gas breakdown data available.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {title && (
        <Typography variant="subtitle2" gutterBottom>
          {title}
        </Typography>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={innerRadius}
            outerRadius={outerRadius}
            paddingAngle={2}
            label={({ name, percent }) =>
              `${name} ${(percent * 100).toFixed(1)}%`
            }
            labelLine={{ strokeWidth: 1 }}
          >
            {chartData.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: number, name: string) => [
              `${formatNumber(value, 2)} tCO2e (${((value / total) * 100).toFixed(1)}%)`,
              name,
            ]}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default GasBreakdownChart;
