import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface RevenueSizingProps { data: { type: string; low: number; mid: number; high: number }[]; }

const RevenueSizing: React.FC<RevenueSizingProps> = ({ data }) => {
  const chartData = data.map((d) => ({ name: d.type.replace(/_/g, ' '), Low: d.low / 1e6, Mid: d.mid / 1e6, High: d.high / 1e6 }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Revenue Opportunity Sizing ($M)</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis tickFormatter={(v) => `$${v}M`} /><Tooltip /><Legend />
          <Bar dataKey="Low" fill="#81C784" radius={[4, 4, 0, 0]} /><Bar dataKey="Mid" fill="#2E7D32" radius={[4, 4, 0, 0]} /><Bar dataKey="High" fill="#1B5E20" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default RevenueSizing;
