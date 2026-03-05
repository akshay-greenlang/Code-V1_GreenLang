/**
 * PCAFQuality - Data quality 1-5 distribution chart.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { PortfolioHolding } from '../../types';

interface PCAFQualityProps { holdings: PortfolioHolding[]; }

const DQ_COLORS = ['#1B5E20', '#4CAF50', '#FFC107', '#EF6C00', '#C62828'];

const PCAFQuality: React.FC<PCAFQualityProps> = ({ holdings }) => {
  const counts = [1, 2, 3, 4, 5].map((dq) => ({ quality: `DQ ${dq}`, count: holdings.filter((h) => h.data_quality === dq).length, dq }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>PCAF Data Quality</Typography>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={counts}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="quality" fontSize={11} />
            <YAxis allowDecimals={false} fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" name="Holdings">
              {counts.map((c, i) => <Cell key={i} fill={DQ_COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <Typography variant="caption" color="text.secondary">DQ 1 = highest quality (verified), DQ 5 = estimated</Typography>
      </CardContent>
    </Card>
  );
};

export default PCAFQuality;
