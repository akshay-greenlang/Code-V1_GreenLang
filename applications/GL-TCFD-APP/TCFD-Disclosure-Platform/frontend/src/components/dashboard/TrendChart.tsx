import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrendChartProps {
  data: { year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[];
}

const TrendChart: React.FC<TrendChartProps> = ({ data }) => {
  const chartData = data.map((d) => ({
    year: d.year.toString(),
    'Scope 1': d.scope_1 / 1000,
    'Scope 2': d.scope_2 / 1000,
    'Scope 3': d.scope_3 / 1000,
    Total: d.total / 1000,
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Emissions Trend (ktCO2e)
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis tickFormatter={(v) => `${v}k`} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}k tCO2e`, '']} />
            <Legend />
            <Line type="monotone" dataKey="Scope 1" stroke="#C62828" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="Scope 2" stroke="#E65100" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="Scope 3" stroke="#F57F17" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="Total" stroke="#1B5E20" strokeWidth={3} dot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default TrendChart;
