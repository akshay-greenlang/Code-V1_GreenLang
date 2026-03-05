/**
 * WACITrend - Weighted Average Carbon Intensity time series chart.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface WACITrendProps { currentWaci: number; trend: { year: number; waci: number }[]; benchmark: number; }

const WACITrend: React.FC<WACITrendProps> = ({ currentWaci, trend, benchmark }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>WACI Trend</Typography>
      <Typography variant="h4" sx={{ fontWeight: 700, color: currentWaci <= benchmark ? '#2E7D32' : '#C62828' }}>
        {currentWaci.toFixed(1)} <Typography component="span" variant="body2">tCO2e/$M</Typography>
      </Typography>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={trend}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" fontSize={11} />
          <YAxis fontSize={11} />
          <Tooltip formatter={(v: number) => [`${v.toFixed(1)} tCO2e/$M`, 'WACI']} />
          <ReferenceLine y={benchmark} stroke="#EF6C00" strokeDasharray="4 4" label={{ value: 'Benchmark', fontSize: 10 }} />
          <Line type="monotone" dataKey="waci" stroke="#0D47A1" strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default WACITrend;
