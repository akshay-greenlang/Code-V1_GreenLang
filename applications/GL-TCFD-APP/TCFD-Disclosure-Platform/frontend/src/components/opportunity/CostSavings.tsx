import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

interface CostSavingsProps { data: { category: string; current_cost: number; savings: number; investment: number }[]; }

const CostSavings: React.FC<CostSavingsProps> = ({ data }) => {
  const chartData = data.map((d) => ({ name: d.category, savings: d.savings / 1e6, investment: -(d.investment / 1e6), net: (d.savings - d.investment) / 1e6 }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Cost Savings Waterfall ($M)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis tickFormatter={(v) => `$${v}M`} />
          <Tooltip formatter={(v: number) => [`$${Math.abs(Number(v)).toFixed(1)}M`, '']} /><Legend /><ReferenceLine y={0} stroke="#000" />
          <Bar dataKey="savings" name="Savings" fill="#2E7D32" radius={[4, 4, 0, 0]} />
          <Bar dataKey="investment" name="Investment" fill="#C62828" radius={[0, 0, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default CostSavings;
