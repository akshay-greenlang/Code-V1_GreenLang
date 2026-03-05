/**
 * GL-ISO14064-APP v1.0 - Emissions vs Removals Waterfall Chart
 *
 * Horizontal bar chart comparing gross emissions, removals, and net
 * emissions for a quick visual balance per ISO 14064-1 Clause 5.
 */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';

interface Props {
  grossEmissions: number;
  totalRemovals: number;
  biogenicCO2: number;
  title?: string;
}

const EmissionsRemovalsChart: React.FC<Props> = ({
  grossEmissions,
  totalRemovals,
  biogenicCO2,
  title = 'Emissions vs Removals',
}) => {
  const theme = useTheme();

  const netEmissions = grossEmissions - totalRemovals;

  const data = [
    { name: 'Gross Emissions', emissions: grossEmissions, removals: 0, net: 0, biogenic: 0 },
    { name: 'Removals', emissions: 0, removals: totalRemovals, net: 0, biogenic: 0 },
    { name: 'Net Emissions', emissions: 0, removals: 0, net: netEmissions, biogenic: 0 },
    { name: 'Biogenic CO2', emissions: 0, removals: 0, net: 0, biogenic: biogenicCO2 },
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        {grossEmissions === 0 && totalRemovals === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={260}>
            <Typography color="text.secondary">No emissions data available</Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 80, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis
                type="number"
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) =>
                  v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`
                }
              />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tCO2e`,
                  name,
                ]}
              />
              <ReferenceLine x={0} stroke={theme.palette.divider} />
              <Bar dataKey="emissions" name="Gross Emissions" fill="#e53935" radius={[0, 4, 4, 0]} />
              <Bar dataKey="removals" name="Removals" fill="#00c853" radius={[0, 4, 4, 0]} />
              <Bar dataKey="net" name="Net Emissions" fill="#1e88e5" radius={[0, 4, 4, 0]} />
              <Bar dataKey="biogenic" name="Biogenic CO2" fill="#78909c" radius={[0, 4, 4, 0]} />
              <Legend verticalAlign="bottom" height={36} iconSize={10} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default EmissionsRemovalsChart;
