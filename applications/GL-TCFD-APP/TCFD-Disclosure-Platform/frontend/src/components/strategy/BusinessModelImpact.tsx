import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { BusinessModelImpact as BMI } from '../../types';
import { formatCompactNumber } from '../../utils/formatters';

interface BusinessModelImpactProps {
  impacts: BMI[];
}

const BusinessModelImpact: React.FC<BusinessModelImpactProps> = ({ impacts }) => {
  const chartData = impacts.map((i) => ({
    area: i.area,
    riskImpact: -Math.abs(i.risk_impacts.reduce((sum, r) => sum + r.impact_value, 0)) / 1_000_000,
    opportunityImpact: i.opportunity_impacts.reduce((sum, o) => sum + o.impact_value, 0) / 1_000_000,
    netImpact: i.net_impact / 1_000_000,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Business Model Impact Analysis ($M)</Typography>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData} stackOffset="sign">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="area" tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(v) => `$${v}M`} />
            <Tooltip formatter={(v: number) => [`$${v.toFixed(1)}M`, '']} />
            <Legend />
            <ReferenceLine y={0} stroke="#000" />
            <Bar dataKey="riskImpact" name="Risk Impact" fill="#C62828" stackId="stack" radius={[0, 0, 0, 0]} />
            <Bar dataKey="opportunityImpact" name="Opportunity Impact" fill="#2E7D32" stackId="stack" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        {impacts.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
            No business model impact data available
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default BusinessModelImpact;
