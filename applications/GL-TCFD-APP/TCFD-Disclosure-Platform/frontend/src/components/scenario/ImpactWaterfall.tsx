import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import type { ScenarioResult } from '../../types';
import { buildWaterfallData } from '../../utils/scenarioHelpers';

interface ImpactWaterfallProps {
  result: ScenarioResult | null;
}

const ImpactWaterfall: React.FC<ImpactWaterfallProps> = ({ result }) => {
  if (!result) return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600 }}>Impact Waterfall</Typography>
      <Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>Run a scenario to see the waterfall chart</Typography>
    </CardContent></Card>
  );

  const data = buildWaterfallData(result);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Financial Impact Waterfall ($M) - {result.scenario_name}
        </Typography>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
            <YAxis tickFormatter={(v) => `$${(v / 1_000_000).toFixed(0)}M`} />
            <Tooltip formatter={(v: number) => [`$${(v / 1_000_000).toFixed(1)}M`, '']} />
            <ReferenceLine y={0} stroke="#000" />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={index} fill={entry.value >= 0 ? '#2E7D32' : '#C62828'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ImpactWaterfall;
