/**
 * PortfolioCoverage - Coverage % with path to 100% by 2040.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import ScoreGauge from '../common/ScoreGauge';

interface PortfolioCoverageProps { overallPct: number; pathTo100: { year: number; coverage_pct: number }[]; }

const PortfolioCoverage: React.FC<PortfolioCoverageProps> = ({ overallPct, pathTo100 }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Portfolio SBTi Coverage</Typography>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <ScoreGauge value={overallPct} label="Current Coverage" subtitle="Target: 100% by 2040" size={120} />
      </Box>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={pathTo100}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" fontSize={10} />
          <YAxis domain={[0, 100]} fontSize={10} />
          <Tooltip formatter={(v: number) => [`${v.toFixed(0)}%`, 'Coverage']} />
          <ReferenceLine y={100} stroke="#2E7D32" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="coverage_pct" stroke="#0D47A1" strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default PortfolioCoverage;
