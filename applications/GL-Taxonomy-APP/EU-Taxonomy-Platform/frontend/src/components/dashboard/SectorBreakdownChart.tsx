/**
 * SectorBreakdownChart - Pie chart showing activities by sector.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const DEMO_DATA = [
  { sector: 'Energy', count: 18, turnover: 245000000, color: '#1B5E20' },
  { sector: 'Manufacturing', count: 14, turnover: 180000000, color: '#0D47A1' },
  { sector: 'Transport', count: 8, turnover: 95000000, color: '#E65100' },
  { sector: 'Construction', count: 6, turnover: 72000000, color: '#4A148C' },
  { sector: 'ICT', count: 4, turnover: 55000000, color: '#01579B' },
  { sector: 'Other', count: 3, turnover: 35000000, color: '#757575' },
];

interface SectorBreakdownChartProps {
  data?: typeof DEMO_DATA;
}

const SectorBreakdownChart: React.FC<SectorBreakdownChartProps> = ({ data = DEMO_DATA }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Sector Breakdown
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            outerRadius={90}
            dataKey="count"
            nameKey="sector"
            label={({ sector, percent }) => `${sector} ${(percent * 100).toFixed(0)}%`}
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

export default SectorBreakdownChart;
