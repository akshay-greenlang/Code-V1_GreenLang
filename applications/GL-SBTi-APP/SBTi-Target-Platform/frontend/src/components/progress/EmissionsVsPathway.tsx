/**
 * EmissionsVsPathway - Line chart of actual emissions vs expected pathway.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ComposedChart } from 'recharts';
import type { ProgressRecord } from '../../types';

interface EmissionsVsPathwayProps { records: ProgressRecord[]; }

const EmissionsVsPathway: React.FC<EmissionsVsPathwayProps> = ({ records }) => {
  const data = records.map((r) => ({ year: r.reporting_year, actual: r.actual_emissions, expected: r.expected_emissions }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Actual vs Target Pathway</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" fontSize={11} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
            <Legend />
            <Area type="monotone" dataKey="expected" fill="#0D47A120" stroke="#0D47A1" strokeWidth={2} strokeDasharray="6 4" name="Target Pathway" />
            <Line type="monotone" dataKey="actual" stroke="#1B5E20" strokeWidth={2.5} name="Actual" dot={{ r: 4 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default EmissionsVsPathway;
