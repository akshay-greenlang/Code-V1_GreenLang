import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, ReferenceLine, Cell } from 'recharts';

interface PriorityMatrixProps { data: { id: string; name: string; impact: number; feasibility: number; size: number; type: string }[]; }

const TYPE_COLORS: Record<string, string> = { resource_efficiency: '#1B5E20', energy_source: '#0D47A1', products_services: '#E65100', markets: '#6A1B9A', resilience: '#00838F' };

const PriorityMatrix: React.FC<PriorityMatrixProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Priority Matrix (Impact vs. Feasibility)</Typography>
    <ResponsiveContainer width="100%" height={350}>
      <ScatterChart margin={{ bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey="feasibility" name="Feasibility" domain={[0, 100]} label={{ value: 'Feasibility', position: 'bottom' }} />
        <YAxis type="number" dataKey="impact" name="Impact" domain={[0, 100]} label={{ value: 'Impact', angle: -90, position: 'insideLeft' }} />
        <ZAxis type="number" dataKey="size" range={[80, 500]} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v: number) => [`${Number(v).toFixed(0)}`, '']}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return <Box sx={{ p: 1, bgcolor: 'white', border: '1px solid #E0E0E0', borderRadius: 1 }}>
              <Typography variant="subtitle2">{d.name}</Typography>
              <Typography variant="caption">Impact: {d.impact} | Feasibility: {d.feasibility}</Typography>
            </Box>;
          }} />
        <ReferenceLine x={50} stroke="#999" strokeDasharray="3 3" />
        <ReferenceLine y={50} stroke="#999" strokeDasharray="3 3" />
        <Scatter data={data}>
          {data.map((entry, i) => <Cell key={i} fill={TYPE_COLORS[entry.type] || '#757575'} />)}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, flexWrap: 'wrap' }}>
      {Object.entries(TYPE_COLORS).map(([type, color]) => (
        <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: color }} />
          <Typography variant="caption">{type.replace(/_/g, ' ')}</Typography>
        </Box>
      ))}
    </Box>
  </CardContent></Card>
);

export default PriorityMatrix;
