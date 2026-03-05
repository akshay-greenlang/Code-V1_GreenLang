import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { InsuranceCostProjection } from '../../types';

interface InsuranceCostProps { data: InsuranceCostProjection[]; }

const InsuranceCost: React.FC<InsuranceCostProps> = ({ data }) => {
  const scenarios = [...new Set(data.map((d) => d.scenario))];
  const years = [...new Set(data.map((d) => d.year))].sort();
  const chartData = years.map((year) => {
    const pt: Record<string, number | string> = { year: year.toString() };
    scenarios.forEach((s) => { const m = data.find((d) => d.year === year && d.scenario === s); pt[s] = m?.projected_premium ? m.projected_premium / 1e6 : 0; });
    const baseline = data.find((d) => d.year === year);
    pt['Baseline'] = baseline?.baseline_premium ? baseline.baseline_premium / 1e6 : 0;
    return pt;
  });
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Insurance Cost Projections ($M)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" /><YAxis tickFormatter={(v) => `$${v}M`} />
          <Tooltip formatter={(v: number) => [`$${Number(v).toFixed(1)}M`, '']} /><Legend />
          <Line type="monotone" dataKey="Baseline" stroke="#757575" strokeWidth={2} strokeDasharray="5 5" />
          {scenarios.map((s, i) => <Line key={s} type="monotone" dataKey={s} stroke={['#C62828', '#E65100', '#0D47A1'][i % 3]} strokeWidth={2} />)}
        </LineChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default InsuranceCost;
