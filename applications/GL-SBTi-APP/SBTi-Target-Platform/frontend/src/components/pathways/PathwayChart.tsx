/**
 * PathwayChart - Line chart with pathway curve + milestones using Recharts.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import type { PathwayMilestone } from '../../types';

interface PathwayChartProps {
  milestones: PathwayMilestone[];
  baseYear: number;
  targetYear: number;
  targetEmissions: number;
  title?: string;
}

const PathwayChart: React.FC<PathwayChartProps> = ({ milestones, baseYear, targetYear, targetEmissions, title = 'Emissions Pathway' }) => {
  const chartData = milestones.map((m) => ({
    year: m.year,
    expected: m.expected_emissions,
    actual: m.actual_emissions,
    budget: m.cumulative_budget,
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>{title}</Typography>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" fontSize={11} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
            <Legend />
            <Line type="monotone" dataKey="expected" stroke="#0D47A1" strokeWidth={2} name="Target Pathway" strokeDasharray="8 4" dot={false} />
            <Line type="monotone" dataKey="actual" stroke="#1B5E20" strokeWidth={2.5} name="Actual Emissions" dot={{ r: 4 }} connectNulls />
            <ReferenceLine y={targetEmissions} stroke="#C62828" strokeDasharray="4 4" label={{ value: `Target: ${targetEmissions.toLocaleString()}`, position: 'right', fontSize: 10 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PathwayChart;
