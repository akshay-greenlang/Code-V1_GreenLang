import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { DashboardSummary } from '../../types';

interface RiskOpportunityChartProps {
  riskExposure: DashboardSummary['risk_exposure'];
  opportunityValue: DashboardSummary['opportunity_value'];
}

const RiskOpportunityChart: React.FC<RiskOpportunityChartProps> = ({ riskExposure, opportunityValue }) => {
  const data = [
    {
      name: 'Physical Risk',
      value: riskExposure.physical_risk_total / 1_000_000,
      fill: '#C62828',
    },
    {
      name: 'Transition Risk',
      value: riskExposure.transition_risk_total / 1_000_000,
      fill: '#E65100',
    },
    {
      name: 'Opportunities',
      value: opportunityValue.total_opportunity_value / 1_000_000,
      fill: '#2E7D32',
    },
    {
      name: 'Cost Savings',
      value: opportunityValue.total_cost_savings / 1_000_000,
      fill: '#1B5E20',
    },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Risk vs. Opportunity Exposure ($M)
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(v) => `$${v}M`} />
            <YAxis type="category" dataKey="name" width={120} />
            <Tooltip formatter={(v: number) => [`$${v.toFixed(1)}M`, 'Value']} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default RiskOpportunityChart;
