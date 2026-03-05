/**
 * TimelineTrend - Area chart showing KPI trends over time.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const DEMO_DATA = [
  { period: 'Q1 2024', turnover_aligned: 35.2, capex_aligned: 42.1, opex_aligned: 30.5 },
  { period: 'Q2 2024', turnover_aligned: 37.8, capex_aligned: 44.6, opex_aligned: 32.1 },
  { period: 'Q3 2024', turnover_aligned: 39.1, capex_aligned: 47.2, opex_aligned: 34.8 },
  { period: 'Q4 2024', turnover_aligned: 40.5, capex_aligned: 49.8, opex_aligned: 36.2 },
  { period: 'Q1 2025', turnover_aligned: 42.5, capex_aligned: 51.3, opex_aligned: 38.7 },
];

interface TimelineTrendProps {
  data?: typeof DEMO_DATA;
}

const TimelineTrend: React.FC<TimelineTrendProps> = ({ data = DEMO_DATA }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Alignment Trend
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="period" />
          <YAxis unit="%" />
          <Tooltip formatter={(val: number) => `${val.toFixed(1)}%`} />
          <Legend />
          <Area type="monotone" dataKey="turnover_aligned" name="Turnover" stroke="#1B5E20" fill="#C8E6C9" strokeWidth={2} />
          <Area type="monotone" dataKey="capex_aligned" name="CapEx" stroke="#0D47A1" fill="#BBDEFB" strokeWidth={2} />
          <Area type="monotone" dataKey="opex_aligned" name="OpEx" stroke="#E65100" fill="#FFE0B2" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default TimelineTrend;
