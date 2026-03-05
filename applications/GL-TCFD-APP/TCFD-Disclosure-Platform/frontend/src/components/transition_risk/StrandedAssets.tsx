import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface StrandedAssetsProps { data: { asset: string; book_value: number; stranded_value: number; timeline: string }[]; }

const StrandedAssets: React.FC<StrandedAssetsProps> = ({ data }) => {
  const chartData = data.map((d) => ({ name: d.asset, bookValue: d.book_value / 1e6, strandedValue: d.stranded_value / 1e6, strandedPct: d.book_value > 0 ? (d.stranded_value / d.book_value * 100) : 0 }));
  const totalBook = data.reduce((sum, d) => sum + d.book_value, 0);
  const totalStranded = data.reduce((sum, d) => sum + d.stranded_value, 0);
  return (
    <Card><CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Stranded Asset Analysis</Typography>
        <Box sx={{ textAlign: 'right' }}>
          <Typography variant="body2" color="text.secondary">Total at Risk</Typography>
          <Typography variant="h5" sx={{ fontWeight: 700, color: 'error.main' }}>${(totalStranded / 1e6).toFixed(0)}M</Typography>
          <Typography variant="caption" color="text.secondary">of ${(totalBook / 1e6).toFixed(0)}M book value ({totalBook > 0 ? (totalStranded / totalBook * 100).toFixed(1) : 0}%)</Typography>
        </Box>
      </Box>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis tickFormatter={(v) => `$${v}M`} /><Tooltip /><Legend />
          <Bar dataKey="bookValue" name="Book Value ($M)" fill="#0D47A1" radius={[4, 4, 0, 0]} />
          <Bar dataKey="strandedValue" name="Stranded Value ($M)" fill="#C62828" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default StrandedAssets;
