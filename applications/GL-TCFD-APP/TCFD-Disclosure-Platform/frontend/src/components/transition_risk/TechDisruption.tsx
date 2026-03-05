import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { TechnologyRisk } from '../../types';

interface TechDisruptionProps { data: TechnologyRisk[]; }

const COLORS = ['#1B5E20', '#0D47A1', '#E65100', '#C62828', '#6A1B9A'];

const TechDisruption: React.FC<TechDisruptionProps> = ({ data }) => {
  const years = Array.from({ length: 11 }, (_, i) => 2025 + i * 3);
  const chartData = years.map((year) => {
    const pt: Record<string, number | string> = { year: year.toString() };
    data.forEach((tech) => {
      const progress = Math.min(100, tech.current_adoption_pct + ((tech.projected_adoption_pct - tech.current_adoption_pct) * (year - 2025)) / (tech.adoption_timeline_years || 1));
      pt[tech.technology] = Math.max(0, Math.min(100, progress));
    });
    return pt;
  });
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Technology Adoption S-Curves (%)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" />
          <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} /><Tooltip formatter={(v: number) => [`${Number(v).toFixed(1)}%`, '']} /><Legend />
          {data.map((tech, i) => <Line key={tech.id} type="basis" dataKey={tech.technology} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 3 }} />)}
        </LineChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default TechDisruption;
