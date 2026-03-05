import React from 'react';
import { Card, CardContent, Typography, Grid, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { MonteCarloResult } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface MonteCarloChartProps { result: MonteCarloResult | null; }

const MonteCarloChart: React.FC<MonteCarloChartProps> = ({ result }) => {
  if (!result) return <Card><CardContent><Typography variant="h6" sx={{ fontWeight: 600 }}>Monte Carlo Simulation</Typography><Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>Run a simulation to see results</Typography></CardContent></Card>;
  const chartData = result.distribution.map((d) => ({ range: `${(d.bin_start / 1e6).toFixed(0)}`, frequency: d.frequency }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Monte Carlo Distribution ({result.iterations.toLocaleString()} iterations)</Typography>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {[{ label: 'Mean', value: result.mean }, { label: 'Median', value: result.median }, { label: 'P5', value: result.p5 }, { label: 'P95', value: result.p95 }].map(({ label, value }) => (
          <Grid item xs={3} key={label}><Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>{formatCurrency(value, 'USD', true)}</Typography>
          </Box></Grid>
        ))}
      </Grid>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" label={{ value: 'Impact ($M)', position: 'bottom' }} tick={{ fontSize: 10 }} />
          <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
          <Tooltip /><ReferenceLine x={`${(result.mean / 1e6).toFixed(0)}`} stroke="#C62828" strokeDasharray="5 5" label="Mean" />
          <Bar dataKey="frequency" fill="#0D47A1" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default MonteCarloChart;
