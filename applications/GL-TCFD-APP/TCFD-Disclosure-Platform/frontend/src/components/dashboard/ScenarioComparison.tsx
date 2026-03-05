import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getScenarioColor } from '../../utils/scenarioHelpers';

interface ScenarioComparisonProps {
  scenarioResults: { name: string; net_impact: number }[];
}

const ScenarioComparison: React.FC<ScenarioComparisonProps> = ({ scenarioResults }) => {
  const data = scenarioResults.map((s) => ({
    name: s.name.length > 18 ? s.name.slice(0, 18) + '...' : s.name,
    fullName: s.name,
    impact: s.net_impact / 1_000_000,
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Scenario Impact Comparison ($M)
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} margin={{ bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-20} textAnchor="end" height={60} tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(v) => `$${v}M`} />
            <Tooltip
              formatter={(v: number) => [`$${v.toFixed(1)}M`, 'Net Impact']}
              labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ''}
            />
            <Bar
              dataKey="impact"
              radius={[4, 4, 0, 0]}
              fill="#0D47A1"
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ScenarioComparison;
