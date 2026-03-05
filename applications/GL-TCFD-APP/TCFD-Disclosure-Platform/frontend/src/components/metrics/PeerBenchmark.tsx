import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { PeerBenchmarkData } from '../../types';

interface PeerBenchmarkProps { data: PeerBenchmarkData[]; }

const PeerBenchmark: React.FC<PeerBenchmarkProps> = ({ data }) => {
  if (data.length === 0) return <Card><CardContent><Typography variant="h6" sx={{ fontWeight: 600 }}>Peer Benchmarking</Typography><Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>No peer data available</Typography></CardContent></Card>;
  const first = data[0];
  const chartData = [
    { name: 'Our Org', value: first.organization_value, fill: '#1B5E20' },
    ...first.peer_values.map((p) => ({ name: p.name, value: p.value, fill: '#90CAF9' })),
    { name: 'Industry Avg', value: first.industry_average, fill: '#757575' },
    { name: 'Best in Class', value: first.best_in_class, fill: '#FDD835' },
  ];
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Peer Benchmark: {first.metric_name} ({first.unit})</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 11 }} />
          <YAxis /><Tooltip /><Bar dataKey="value" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default PeerBenchmark;
