/**
 * ScoreHistogram - Score distribution bar chart
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ScoreDistributionBucket } from '../../types';
import { SCORING_LEVEL_COLORS } from '../../types';

interface ScoreHistogramProps { distribution: ScoreDistributionBucket[]; }

const ScoreHistogram: React.FC<ScoreHistogramProps> = ({ distribution }) => {
  const data = distribution.map((d) => ({
    name: d.level, count: d.count, pct: d.percentage,
    color: SCORING_LEVEL_COLORS[d.level] || '#9e9e9e',
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Score Distribution</Typography>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value: number, name: string) => [name === 'count' ? value : `${value.toFixed(1)}%`, name === 'count' ? 'Companies' : 'Percentage']} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.map((entry, idx) => (<Cell key={idx} fill={entry.color} />))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ScoreHistogram;
