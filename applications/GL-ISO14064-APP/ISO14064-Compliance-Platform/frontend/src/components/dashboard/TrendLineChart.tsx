/**
 * GL-ISO14064-APP v1.0 - Emissions Trend Line Chart
 *
 * Stacked-area / multi-line chart showing emission trends over time
 * across the six ISO 14064-1 categories with optional removals overlay.
 */

import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';
import type { TrendDataPoint } from '../../types';
import { CATEGORY_COLORS, ISOCategory } from '../../types';

interface Props {
  data: TrendDataPoint[];
  title?: string;
  showRemovals?: boolean;
}

const SERIES = [
  { key: 'category_1_tco2e', label: 'Cat 1 - Direct', color: CATEGORY_COLORS[ISOCategory.CATEGORY_1_DIRECT] },
  { key: 'category_2_tco2e', label: 'Cat 2 - Energy', color: CATEGORY_COLORS[ISOCategory.CATEGORY_2_ENERGY] },
  { key: 'category_3_tco2e', label: 'Cat 3 - Transport', color: CATEGORY_COLORS[ISOCategory.CATEGORY_3_TRANSPORT] },
  { key: 'category_4_tco2e', label: 'Cat 4 - Products Used', color: CATEGORY_COLORS[ISOCategory.CATEGORY_4_PRODUCTS_USED] },
  { key: 'category_5_tco2e', label: 'Cat 5 - Products From Org', color: CATEGORY_COLORS[ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG] },
  { key: 'category_6_tco2e', label: 'Cat 6 - Other', color: CATEGORY_COLORS[ISOCategory.CATEGORY_6_OTHER] },
];

const TrendLineChart: React.FC<Props> = ({ data, title = 'Emissions Trend', showRemovals = true }) => {
  const theme = useTheme();

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        {data.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={280}>
            <Typography color="text.secondary">No trend data available</Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height={340}>
            <AreaChart data={data} margin={{ top: 8, right: 24, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis dataKey="period_label" tick={{ fontSize: 11 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) =>
                  v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`
                }
                label={{ value: 'tCO2e', angle: -90, position: 'insideLeft', offset: -2, fontSize: 11 }}
              />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tCO2e`,
                  name,
                ]}
              />
              <Legend verticalAlign="bottom" height={40} iconSize={10} />
              {SERIES.map((s) => (
                <Area
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.label}
                  stackId="categories"
                  stroke={s.color}
                  fill={s.color}
                  fillOpacity={0.45}
                />
              ))}
              {showRemovals && (
                <Area
                  type="monotone"
                  dataKey="removals_tco2e"
                  name="Removals"
                  stroke="#00c853"
                  fill="#00c853"
                  fillOpacity={0.2}
                  strokeDasharray="5 3"
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default TrendLineChart;
