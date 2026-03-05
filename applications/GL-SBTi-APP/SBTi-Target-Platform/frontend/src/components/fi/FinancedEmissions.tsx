/**
 * FinancedEmissions - By asset class bar chart.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface FinancedEmissionsProps { byAssetClass: { asset_class: string; emissions: number; pct: number }[]; total: number; }

const COLORS = ['#1B5E20', '#0D47A1', '#4A148C', '#EF6C00', '#C62828', '#006064', '#33691E', '#880E4F'];

const FinancedEmissions: React.FC<FinancedEmissionsProps> = ({ byAssetClass, total }) => {
  const data = byAssetClass.map((a) => ({ ...a, name: a.asset_class.replace(/_/g, ' ') }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>Financed Emissions</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Total: {total.toLocaleString()} tCO2e</Typography>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={10} />
            <YAxis type="category" dataKey="name" fontSize={10} width={120} />
            <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', 'Emissions']} />
            <Bar dataKey="emissions" name="Financed Emissions">
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default FinancedEmissions;
