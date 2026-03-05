import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

interface CarbonSensitivityProps { data: { carbon_price: number; financial_impact: number; scenario: string }[]; }

const COLORS = ['#1B5E20', '#E65100', '#C62828', '#0D47A1'];

const CarbonSensitivity: React.FC<CarbonSensitivityProps> = ({ data }) => {
  const scenarios = [...new Set(data.map((d) => d.scenario))];
  const prices = [...new Set(data.map((d) => d.carbon_price))].sort((a, b) => a - b);
  const chartData = prices.map((price) => {
    const pt: Record<string, number | string> = { price: `$${price}` };
    scenarios.forEach((s) => { const m = data.find((d) => d.carbon_price === price && d.scenario === s); pt[s] = m ? m.financial_impact / 1e6 : 0; });
    return pt;
  });
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Carbon Price Sensitivity ($M Impact)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="price" /><YAxis tickFormatter={(v) => `$${v}M`} />
          <Tooltip /><Legend /><ReferenceLine y={0} stroke="#000" />
          {scenarios.map((s, i) => <Line key={s} type="monotone" dataKey={s} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 3 }} />)}
        </LineChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default CarbonSensitivity;
