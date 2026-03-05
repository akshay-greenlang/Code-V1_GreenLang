import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { MarketRisk } from '../../types';

interface MarketShiftProps { data: MarketRisk[]; }

const MarketShift: React.FC<MarketShiftProps> = ({ data }) => {
  const chartData = data.map((r) => ({
    segment: r.market_segment,
    demandChange: r.demand_change_pct,
    revenueAtRisk: r.revenue_at_risk / 1e6,
    opportunity: r.market_opportunity / 1e6,
  }));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Market Demand Shifts</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="segment" tick={{ fontSize: 11 }} /><YAxis tickFormatter={(v) => `$${v}M`} />
          <Tooltip /><Legend /><ReferenceLine y={0} stroke="#000" />
          <Bar dataKey="revenueAtRisk" name="Revenue at Risk ($M)" fill="#C62828" radius={[4, 4, 0, 0]} />
          <Bar dataKey="opportunity" name="Market Opportunity ($M)" fill="#2E7D32" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent></Card>
  );
};

export default MarketShift;
