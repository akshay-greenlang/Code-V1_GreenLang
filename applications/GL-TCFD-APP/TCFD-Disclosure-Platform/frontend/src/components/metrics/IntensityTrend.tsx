import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface IntensityTrendProps { data: { year: number; revenue_intensity: number; employee_intensity: number }[]; }

const IntensityTrend: React.FC<IntensityTrendProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Emissions Intensity Trends</Typography>
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data.map((d) => ({ ...d, year: d.year.toString() }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" />
        <YAxis /><Tooltip /><Legend />
        <Line type="monotone" dataKey="revenue_intensity" name="Revenue Intensity (tCO2e/$M)" stroke="#1B5E20" strokeWidth={2} dot={{ r: 4 }} />
        <Line type="monotone" dataKey="employee_intensity" name="Employee Intensity (tCO2e/FTE)" stroke="#0D47A1" strokeWidth={2} dot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  </CardContent></Card>
);

export default IntensityTrend;
