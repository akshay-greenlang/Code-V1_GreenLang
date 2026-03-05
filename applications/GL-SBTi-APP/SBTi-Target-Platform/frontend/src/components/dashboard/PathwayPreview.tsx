/**
 * PathwayPreview - Mini pathway chart with actuals overlay for dashboard.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface PathwayPreviewProps {
  emissionsTrend: { year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[];
}

const PathwayPreview: React.FC<PathwayPreviewProps> = ({ emissionsTrend }) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Emissions Trend</Typography>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={emissionsTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" fontSize={11} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
            <Legend />
            <Line type="monotone" dataKey="scope_1" stroke="#1B5E20" strokeWidth={2} name="Scope 1" dot={false} />
            <Line type="monotone" dataKey="scope_2" stroke="#0D47A1" strokeWidth={2} name="Scope 2" dot={false} />
            <Line type="monotone" dataKey="scope_3" stroke="#EF6C00" strokeWidth={2} name="Scope 3" dot={false} />
            <Line type="monotone" dataKey="total" stroke="#B71C1C" strokeWidth={2} strokeDasharray="5 5" name="Total" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PathwayPreview;
