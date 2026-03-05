/**
 * ProjectionChart - Future projection to target year.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';

interface ProjectionChartProps { projection: { year: number; projected_emissions: number; target_emissions: number }[]; targetYear: number; }

const ProjectionChart: React.FC<ProjectionChartProps> = ({ projection, targetYear }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Future Projection</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={projection}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" fontSize={11} />
          <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
          <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
          <Legend />
          <Line type="monotone" dataKey="projected_emissions" stroke="#EF6C00" strokeWidth={2} strokeDasharray="8 4" name="Projected" dot={false} />
          <Line type="monotone" dataKey="target_emissions" stroke="#0D47A1" strokeWidth={2} name="Target Pathway" dot={false} />
          <ReferenceLine x={targetYear} stroke="#C62828" strokeDasharray="4 4" label={{ value: 'Target Year', position: 'top', fontSize: 10 }} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default ProjectionChart;
