/**
 * ScopeTemperature - Per-scope temperature bars.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { getTemperatureColor } from '../../utils/pathwayHelpers';

interface ScopeTemperatureProps { scores: { scope: string; temperature: number }[]; }

const ScopeTemperature: React.FC<ScopeTemperatureProps> = ({ scores }) => {
  const data = scores.map((s) => ({ scope: s.scope.replace(/_/g, ' ').toUpperCase(), temperature: s.temperature }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Temperature by Scope</Typography>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="scope" fontSize={11} />
            <YAxis domain={[0, 4]} fontSize={11} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(2)}\u00B0C`, 'Temperature']} />
            <ReferenceLine y={1.5} stroke="#1B5E20" strokeDasharray="4 4" label={{ value: '1.5\u00B0C', position: 'right', fontSize: 10 }} />
            <ReferenceLine y={2.0} stroke="#EF6C00" strokeDasharray="4 4" label={{ value: '2\u00B0C', position: 'right', fontSize: 10 }} />
            <Bar dataKey="temperature" name="Temperature">
              {data.map((d, i) => <Cell key={i} fill={getTemperatureColor(d.temperature)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ScopeTemperature;
