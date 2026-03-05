import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import type { MACCDataPoint } from '../../types';

interface MACCChartProps { data: MACCDataPoint[]; }

const MACCChart: React.FC<MACCChartProps> = ({ data }) => {
  const sorted = [...data].sort((a, b) => a.cost_per_tco2e - b.cost_per_tco2e);
  let cumAbatement = 0;
  const chartData = sorted.map((d) => {
    const start = cumAbatement;
    cumAbatement += d.abatement_potential_tco2e;
    return { name: d.measure, cost: d.cost_per_tco2e, width: d.abatement_potential_tco2e, start, mid: start + d.abatement_potential_tco2e / 2, status: d.status };
  });
  const STATUS_COLORS: Record<string, string> = { implemented: '#1B5E20', approved: '#2E7D32', evaluating: '#F57F17', identified: '#9E9E9E' };

  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Marginal Abatement Cost Curve (MACC)</Typography>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData} barCategoryGap={0}>
          <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={80} />
          <YAxis label={{ value: '$/tCO2e', angle: -90, position: 'insideLeft' }} /><ReferenceLine y={0} stroke="#000" />
          <Tooltip content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return <Box sx={{ p: 1, bgcolor: 'white', border: '1px solid #E0E0E0', borderRadius: 1 }}>
              <Typography variant="subtitle2">{d.name}</Typography>
              <Typography variant="caption">Cost: ${d.cost.toFixed(0)}/tCO2e</Typography><br/>
              <Typography variant="caption">Abatement: {d.width.toLocaleString()} tCO2e</Typography>
            </Box>;
          }} />
          <Bar dataKey="cost" radius={[2, 2, 0, 0]}>
            {chartData.map((entry, i) => <Cell key={i} fill={entry.cost < 0 ? '#2E7D32' : STATUS_COLORS[entry.status] || '#E65100'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default MACCChart;
