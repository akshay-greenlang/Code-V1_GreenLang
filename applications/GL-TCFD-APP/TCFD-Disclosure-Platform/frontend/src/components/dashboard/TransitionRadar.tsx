import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

interface TransitionRadarProps {
  data: { dimension: string; value: number; fullMark?: number }[];
}

const DEFAULT_DATA = [
  { dimension: 'Policy & Legal', value: 65, fullMark: 100 },
  { dimension: 'Technology', value: 45, fullMark: 100 },
  { dimension: 'Market', value: 70, fullMark: 100 },
  { dimension: 'Reputation', value: 55, fullMark: 100 },
];

const TransitionRadar: React.FC<TransitionRadarProps> = ({ data }) => {
  const chartData = data.length > 0 ? data : DEFAULT_DATA;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
          Transition Risk Profile
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={chartData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Radar
              name="Risk Exposure"
              dataKey="value"
              stroke="#E65100"
              fill="#E65100"
              fillOpacity={0.3}
            />
            <Tooltip formatter={(v: number) => [`${v}%`, 'Exposure']} />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default TransitionRadar;
