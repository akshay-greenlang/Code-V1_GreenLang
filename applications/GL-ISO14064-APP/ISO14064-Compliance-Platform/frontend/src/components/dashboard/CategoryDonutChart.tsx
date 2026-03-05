/**
 * GL-ISO14064-APP v1.0 - ISO 14064-1 Category Donut Chart
 *
 * Renders a donut chart of emissions split across ISO 14064-1's
 * six indirect/direct categories with centre-label net total.
 */

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';
import type { CategoryBreakdownItem } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, CATEGORY_COLORS, ISOCategory } from '../../types';

interface Props {
  data: CategoryBreakdownItem[];
  title?: string;
}

const CategoryDonutChart: React.FC<Props> = ({ data, title = 'Emissions by ISO Category' }) => {
  const theme = useTheme();

  const chartData = data.map((item) => ({
    name: ISO_CATEGORY_SHORT_NAMES[item.category] ?? item.category_name,
    value: Math.abs(item.net_tco2e),
    fill: CATEGORY_COLORS[item.category] ?? theme.palette.grey[400],
    pct: item.percentage_of_total,
  }));

  const netTotal = data.reduce((sum, d) => sum + d.net_tco2e, 0);

  const renderLabel = ({ cx, cy }: { cx: number; cy: number }) => (
    <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central">
      <tspan x={cx} dy="-0.5em" fontSize={14} fontWeight={600} fill={theme.palette.text.primary}>
        {netTotal.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </tspan>
      <tspan x={cx} dy="1.4em" fontSize={11} fill={theme.palette.text.secondary}>
        tCO2e net
      </tspan>
    </text>
  );

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        {chartData.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={280}>
            <Typography color="text.secondary">No category data available</Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={110}
                paddingAngle={2}
                label={false}
              >
                {chartData.map((entry, i) => (
                  <Cell key={`cell-${i}`} fill={entry.fill} strokeWidth={1} />
                ))}
              </Pie>
              {renderLabel({ cx: 200, cy: 160 })}
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value.toLocaleString()} tCO2e`,
                  name,
                ]}
              />
              <Legend verticalAlign="bottom" height={48} iconSize={10} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default CategoryDonutChart;
