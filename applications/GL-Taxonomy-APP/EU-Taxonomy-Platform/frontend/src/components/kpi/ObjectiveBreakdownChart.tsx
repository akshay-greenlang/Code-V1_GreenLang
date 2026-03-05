/**
 * ObjectiveBreakdownChart - Stacked bar chart of KPIs by objective.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const DEMO_DATA = [
  { objective: 'CCM', turnover: 35.2, capex: 42.1, opex: 30.5 },
  { objective: 'CCA', turnover: 4.8, capex: 6.2, opex: 5.1 },
  { objective: 'WTR', turnover: 1.2, capex: 1.5, opex: 1.8 },
  { objective: 'CE', turnover: 0.8, capex: 1.1, opex: 0.9 },
  { objective: 'PPC', turnover: 0.3, capex: 0.2, opex: 0.2 },
  { objective: 'BIO', turnover: 0.2, capex: 0.2, opex: 0.2 },
];

const ObjectiveBreakdownChart: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Aligned KPI by Objective
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={DEMO_DATA}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="objective" />
          <YAxis unit="%" />
          <Tooltip formatter={(val: number) => `${val.toFixed(1)}%`} />
          <Legend />
          <Bar dataKey="turnover" name="Turnover" fill="#1B5E20" />
          <Bar dataKey="capex" name="CapEx" fill="#0D47A1" />
          <Bar dataKey="opex" name="OpEx" fill="#E65100" />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default ObjectiveBreakdownChart;
