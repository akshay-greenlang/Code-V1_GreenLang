/**
 * AlignmentOverview - Donut chart showing aligned vs eligible vs non-eligible.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const DEMO_DATA = [
  { name: 'Aligned', value: 42, color: '#1B5E20' },
  { name: 'Eligible (not aligned)', value: 23, color: '#4C8C4A' },
  { name: 'Not Eligible', value: 35, color: '#BDBDBD' },
];

interface AlignmentOverviewProps {
  data?: { name: string; value: number; color: string }[];
}

const AlignmentOverview: React.FC<AlignmentOverviewProps> = ({ data = DEMO_DATA }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Alignment Overview
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            dataKey="value"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            labelLine={false}
          >
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip formatter={(val: number) => `${val} activities`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default AlignmentOverview;
