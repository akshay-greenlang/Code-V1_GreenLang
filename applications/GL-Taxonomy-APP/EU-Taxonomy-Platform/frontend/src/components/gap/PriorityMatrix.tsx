/**
 * PriorityMatrix - Impact vs Effort scatter chart for gap prioritization.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Cell, ReferenceLine } from 'recharts';

const DEMO_DATA = [
  { name: 'EPC data collection', effort: 30, impact: 85, severity: 'critical' },
  { name: 'Water DNSH assessment', effort: 45, impact: 70, severity: 'high' },
  { name: 'Climate risk CCA', effort: 60, impact: 75, severity: 'high' },
  { name: 'Anti-corruption policy', effort: 20, impact: 50, severity: 'medium' },
  { name: 'SC evidence verification', effort: 70, impact: 40, severity: 'medium' },
  { name: 'NACE mapping review', effort: 15, impact: 25, severity: 'low' },
  { name: 'Reporting template update', effort: 25, impact: 35, severity: 'low' },
];

const colorMap: Record<string, string> = { critical: '#C62828', high: '#E65100', medium: '#EF6C00', low: '#2E7D32' };

const PriorityMatrix: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Priority Matrix (Impact vs Effort)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="effort" name="Effort" domain={[0, 100]} label={{ value: 'Effort', position: 'insideBottom', offset: -5 }} />
          <YAxis type="number" dataKey="impact" name="Impact" domain={[0, 100]} label={{ value: 'Impact', angle: -90, position: 'insideLeft' }} />
          <ReferenceLine x={50} stroke="#E0E0E0" strokeDasharray="3 3" />
          <ReferenceLine y={50} stroke="#E0E0E0" strokeDasharray="3 3" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(val: number, name: string) => [`${val}`, name]} />
          <Scatter data={DEMO_DATA} fill="#1B5E20">
            {DEMO_DATA.map((entry, idx) => (
              <Cell key={idx} fill={colorMap[entry.severity]} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1 }}>
        <Typography variant="caption" color="text.secondary">Top-Left: Quick Wins | Top-Right: Strategic | Bottom-Left: Low Priority | Bottom-Right: Deprioritize</Typography>
      </Box>
    </CardContent>
  </Card>
);

export default PriorityMatrix;
