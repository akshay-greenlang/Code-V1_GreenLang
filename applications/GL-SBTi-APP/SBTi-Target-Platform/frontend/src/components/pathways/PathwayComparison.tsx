/**
 * PathwayComparison - Multi-pathway overlay comparison chart.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { PathwayComparison as PathwayComparisonType } from '../../types';
import { buildComparisonChartData } from '../../utils/pathwayHelpers';
import { formatPathwayAlignment } from '../../utils/formatters';

interface PathwayComparisonProps {
  comparisons: PathwayComparisonType[];
}

const COLORS = ['#0D47A1', '#1B5E20', '#EF6C00', '#7B1FA2', '#C62828'];

const PathwayComparisonComponent: React.FC<PathwayComparisonProps> = ({ comparisons }) => {
  const chartData = buildComparisonChartData(comparisons);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Pathway Comparison</Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          {comparisons.map((c, idx) => (
            <Chip key={c.pathway_id} label={`${c.pathway_name} (${formatPathwayAlignment(c.alignment)} | ${c.annual_rate.toFixed(1)}%/yr)`}
              size="small" sx={{ backgroundColor: COLORS[idx % COLORS.length] + '20', color: COLORS[idx % COLORS.length], fontWeight: 600 }}
            />
          ))}
        </Box>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" fontSize={11} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number) => [value?.toLocaleString() + ' tCO2e', '']} />
            <Legend />
            {comparisons.map((c, idx) => (
              <Line key={c.pathway_id} type="monotone" dataKey={c.pathway_name} stroke={COLORS[idx % COLORS.length]} strokeWidth={2} dot={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PathwayComparisonComponent;
