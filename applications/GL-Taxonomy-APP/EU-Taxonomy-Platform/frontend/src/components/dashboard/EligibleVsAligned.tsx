/**
 * EligibleVsAligned - Funnel visualization from total to aligned.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';

const DEMO_DATA = [
  { stage: 'Total', count: 53, color: '#BDBDBD' },
  { stage: 'Eligible', count: 38, color: '#0277BD' },
  { stage: 'SC Pass', count: 32, color: '#4C8C4A' },
  { stage: 'DNSH Pass', count: 28, color: '#388E3C' },
  { stage: 'MS Pass', count: 26, color: '#2E7D32' },
  { stage: 'Aligned', count: 24, color: '#1B5E20' },
];

interface EligibleVsAlignedProps {
  data?: typeof DEMO_DATA;
}

const EligibleVsAligned: React.FC<EligibleVsAlignedProps> = ({ data = DEMO_DATA }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Alignment Funnel
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" />
          <YAxis type="category" dataKey="stage" width={80} />
          <Tooltip formatter={(val: number) => `${val} activities`} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default EligibleVsAligned;
