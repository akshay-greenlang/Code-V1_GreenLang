import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface HazardChartProps { data: { hazard_type: string; asset_count: number; total_exposure: number }[]; }

const HazardChart: React.FC<HazardChartProps> = ({ data }) => {
  const chartData = data.map((d) => ({ name: d.hazard_type.replace(/_/g, ' '), assets: d.asset_count, exposure: d.total_exposure / 1e6 }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Hazard Distribution</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-25} textAnchor="end" height={80} />
          <YAxis yAxisId="left" tickFormatter={(v) => `${v}`} /><YAxis yAxisId="right" orientation="right" tickFormatter={(v) => `$${v}M`} />
          <Tooltip /><Bar yAxisId="left" dataKey="assets" fill="#0D47A1" name="Assets" radius={[4, 4, 0, 0]} />
          <Bar yAxisId="right" dataKey="exposure" fill="#C62828" name="Exposure ($M)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default HazardChart;
