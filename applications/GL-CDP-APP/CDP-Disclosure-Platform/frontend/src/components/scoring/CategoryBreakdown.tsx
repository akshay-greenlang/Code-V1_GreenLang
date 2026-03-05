/**
 * CategoryBreakdown - 17 category bar chart
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { CategoryScore } from '../../types';
import { SCORING_CATEGORY_NAMES, SCORING_CATEGORY_COLORS, ScoringCategory } from '../../types';

interface CategoryBreakdownProps { categories: CategoryScore[]; }

const CategoryBreakdown: React.FC<CategoryBreakdownProps> = ({ categories }) => {
  const chartData = categories.map((c) => ({
    name: (SCORING_CATEGORY_NAMES[c.category] || c.category).split(' ').slice(0, 2).join(' '),
    fullName: SCORING_CATEGORY_NAMES[c.category] || c.category,
    score: c.percentage,
    color: SCORING_CATEGORY_COLORS[c.category] || '#9e9e9e',
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Category Breakdown</Typography>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={80} />
            <Tooltip formatter={(value: number, _name: string, props: { payload: { fullName: string } }) => [`${value.toFixed(1)}%`, props.payload.fullName]} />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, idx) => (
                <Cell key={idx} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default CategoryBreakdown;
