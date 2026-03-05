import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts';
import type { SensitivityResult } from '../../types';

interface SensitivityTornadoProps {
  data: SensitivityResult[];
}

const SensitivityTornado: React.FC<SensitivityTornadoProps> = ({ data }) => {
  const sorted = [...data].sort((a, b) =>
    Math.abs(b.high_impact - b.low_impact) - Math.abs(a.high_impact - a.low_impact)
  );
  const chartData = sorted.map((r) => ({
    name: r.parameter_name,
    downside: r.low_impact / 1_000_000,
    upside: r.high_impact / 1_000_000,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Sensitivity Analysis (Tornado Chart, $M)</Typography>
        <ResponsiveContainer width="100%" height={Math.max(250, data.length * 40 + 60)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 30 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(v) => `$${v}M`} />
            <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v: number) => [`$${Number(v).toFixed(1)}M`, '']} />
            <Legend />
            <ReferenceLine x={0} stroke="#000" />
            <Bar dataKey="downside" fill="#C62828" name="Downside" />
            <Bar dataKey="upside" fill="#2E7D32" name="Upside" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default SensitivityTornado;
