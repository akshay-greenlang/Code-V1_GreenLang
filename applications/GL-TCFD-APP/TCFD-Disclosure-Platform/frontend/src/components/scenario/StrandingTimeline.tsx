import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { StrandingDataPoint } from '../../types';

interface StrandingTimelineProps {
  data: StrandingDataPoint[];
}

const COLORS = ['#1B5E20', '#E65100', '#C62828', '#0D47A1', '#6A1B9A'];

const StrandingTimeline: React.FC<StrandingTimelineProps> = ({ data }) => {
  const scenarios = [...new Set(data.map((d) => d.scenario_name))];
  const years = [...new Set(data.map((d) => d.year))].sort();
  const chartData = years.map((year) => {
    const point: Record<string, number | string> = { year: year.toString() };
    scenarios.forEach((s) => {
      const match = data.find((d) => d.year === year && d.scenario_name === s);
      point[s] = match?.percentage_at_risk || 0;
    });
    return point;
  });

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Asset Stranding Timeline (% at Risk)</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
            <Tooltip formatter={(v: number) => [`${Number(v).toFixed(1)}%`, '']} />
            <Legend />
            {scenarios.map((s, i) => (
              <Line key={s} type="monotone" dataKey={s} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 3 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default StrandingTimeline;
