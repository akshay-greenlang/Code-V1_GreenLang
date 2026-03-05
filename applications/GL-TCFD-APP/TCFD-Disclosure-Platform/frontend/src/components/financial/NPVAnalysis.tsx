import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { NPVResult } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface NPVAnalysisProps { result: NPVResult | null; }

const NPVAnalysis: React.FC<NPVAnalysisProps> = ({ result }) => {
  if (!result) return <Card><CardContent><Typography variant="h6" sx={{ fontWeight: 600 }}>NPV Analysis</Typography><Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>Select a scenario to view NPV analysis</Typography></CardContent></Card>;
  const chartData = result.cash_flows.map((cf) => ({ year: cf.year.toString(), cashFlow: cf.amount / 1e6 }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>NPV / IRR Analysis - {result.scenario_name}</Typography>
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center' }}><Typography variant="caption" color="text.secondary">NPV</Typography><Typography variant="h5" sx={{ fontWeight: 700, color: result.npv >= 0 ? 'success.main' : 'error.main' }}>{formatCurrency(result.npv, 'USD', true)}</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center' }}><Typography variant="caption" color="text.secondary">IRR</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{(result.irr * 100).toFixed(1)}%</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center' }}><Typography variant="caption" color="text.secondary">Payback</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{result.payback_years.toFixed(1)} yrs</Typography></Box></Grid>
        <Grid item xs={6} sm={3}><Box sx={{ textAlign: 'center' }}><Typography variant="caption" color="text.secondary">Discount Rate</Typography><Typography variant="h5" sx={{ fontWeight: 700 }}>{(result.discount_rate * 100).toFixed(1)}%</Typography></Box></Grid>
      </Grid>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" /><YAxis tickFormatter={(v) => `$${v}M`} />
          <Tooltip formatter={(v: number) => [`$${Number(v).toFixed(1)}M`, 'Cash Flow']} /><ReferenceLine y={0} stroke="#000" />
          <Line type="monotone" dataKey="cashFlow" stroke="#1B5E20" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default NPVAnalysis;
