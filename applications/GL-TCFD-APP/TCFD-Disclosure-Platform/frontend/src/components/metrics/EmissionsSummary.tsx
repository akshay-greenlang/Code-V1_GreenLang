import React from 'react';
import { Card, CardContent, Typography, Box, Grid } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { EmissionsSummary as ES } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface EmissionsSummaryProps { data: ES | null; }

const EmissionsSummaryComponent: React.FC<EmissionsSummaryProps> = ({ data }) => {
  if (!data) return <Card><CardContent><Typography variant="h6" sx={{ fontWeight: 600 }}>Emissions Summary</Typography><Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>No emissions data available</Typography></CardContent></Card>;
  const chartData = [
    { scope: 'Scope 1', value: data.scope_1, color: '#C62828' },
    { scope: 'Scope 2 (Location)', value: data.scope_2_location, color: '#E65100' },
    { scope: 'Scope 2 (Market)', value: data.scope_2_market, color: '#F57F17' },
    { scope: 'Scope 3', value: data.scope_3_total, color: '#0D47A1' },
  ];
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>GHG Emissions Summary ({data.reporting_year})</Typography>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center', p: 1, bgcolor: '#FAFAFA', borderRadius: 1 }}><Typography variant="caption" color="text.secondary">Total Emissions</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{formatNumber(data.total_emissions)}</Typography><Typography variant="caption">tCO2e</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center', p: 1, bgcolor: '#FAFAFA', borderRadius: 1 }}><Typography variant="caption" color="text.secondary">YoY Change</Typography><Typography variant="h5" sx={{ fontWeight: 700, color: data.change_pct <= 0 ? 'success.main' : 'error.main' }}>{data.change_pct > 0 ? '+' : ''}{data.change_pct.toFixed(1)}%</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center', p: 1, bgcolor: '#FAFAFA', borderRadius: 1 }}><Typography variant="caption" color="text.secondary">Revenue Intensity</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{data.intensity_revenue.toFixed(1)}</Typography><Typography variant="caption">tCO2e/$M</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center', p: 1, bgcolor: '#FAFAFA', borderRadius: 1 }}><Typography variant="caption" color="text.secondary">Employee Intensity</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{data.intensity_employee.toFixed(1)}</Typography><Typography variant="caption">tCO2e/FTE</Typography></Box></Grid>
      </Grid>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="scope" />
          <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} /><Tooltip formatter={(v: number) => [`${formatNumber(v)} tCO2e`, '']} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>{chartData.map((entry, i) => <rect key={i} fill={entry.color} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default EmissionsSummaryComponent;
