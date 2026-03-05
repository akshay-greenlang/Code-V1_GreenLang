/**
 * EmissionsSplit - FLAG vs non-FLAG pie chart.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface EmissionsSplitProps { flagEmissions: number; nonFlagEmissions: number; flagPct: number; }

const COLORS = ['#2E7D32', '#0D47A1'];

const EmissionsSplit: React.FC<EmissionsSplitProps> = ({ flagEmissions, nonFlagEmissions, flagPct }) => {
  const data = [
    { name: 'FLAG Emissions', value: flagEmissions },
    { name: 'Non-FLAG Emissions', value: nonFlagEmissions },
  ];
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>FLAG vs Non-FLAG</Typography>
        <Box sx={{ position: 'relative' }}>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={2} dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}>
                {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default EmissionsSplit;
