/**
 * ObjectiveRadar - Radar chart for 6 environmental objectives.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const DEMO_DATA = [
  { objective: 'CCM', eligible: 85, aligned: 62 },
  { objective: 'CCA', eligible: 70, aligned: 45 },
  { objective: 'WTR', eligible: 55, aligned: 30 },
  { objective: 'CE', eligible: 60, aligned: 35 },
  { objective: 'PPC', eligible: 50, aligned: 28 },
  { objective: 'BIO', eligible: 40, aligned: 20 },
];

interface ObjectiveRadarProps {
  data?: typeof DEMO_DATA;
}

const ObjectiveRadar: React.FC<ObjectiveRadarProps> = ({ data = DEMO_DATA }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Objective Coverage
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="objective" />
          <PolarRadiusAxis angle={90} domain={[0, 100]} />
          <Radar name="Eligible %" dataKey="eligible" stroke="#0277BD" fill="#BBDEFB" fillOpacity={0.4} />
          <Radar name="Aligned %" dataKey="aligned" stroke="#1B5E20" fill="#C8E6C9" fillOpacity={0.4} />
          <Legend />
          <Tooltip formatter={(val: number) => `${val}%`} />
        </RadarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
);

export default ObjectiveRadar;
