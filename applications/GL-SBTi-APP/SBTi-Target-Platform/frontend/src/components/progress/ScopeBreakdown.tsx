/**
 * ScopeBreakdown - Stacked bar by scope.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { ProgressRecord } from '../../types';

interface ScopeBreakdownProps { records: ProgressRecord[]; }

const ScopeBreakdown: React.FC<ScopeBreakdownProps> = ({ records }) => {
  const data = records.map((r) => ({ year: r.reporting_year, scope_1: r.scope_1_emissions, scope_2: r.scope_2_emissions, scope_3: r.scope_3_emissions }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Scope Breakdown</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" fontSize={11} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
            <Legend />
            <Bar dataKey="scope_1" stackId="a" fill="#1B5E20" name="Scope 1" />
            <Bar dataKey="scope_2" stackId="a" fill="#0D47A1" name="Scope 2" />
            <Bar dataKey="scope_3" stackId="a" fill="#EF6C00" name="Scope 3" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ScopeBreakdown;
