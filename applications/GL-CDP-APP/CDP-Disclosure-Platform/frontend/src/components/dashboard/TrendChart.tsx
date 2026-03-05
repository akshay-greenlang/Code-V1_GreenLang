/**
 * TrendChart - Year-over-year score trend line chart
 *
 * Displays historical CDP scores over multiple years
 * with band-colored background zones.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts';
import type { HistoricalScore } from '../../types';

interface TrendChartProps {
  scores: HistoricalScore[];
}

const TrendChart: React.FC<TrendChartProps> = ({ scores }) => {
  const chartData = scores.map((s) => ({
    year: s.year.toString(),
    score: s.score,
    level: s.level,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Score Trend
        </Typography>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              {/* Band background zones */}
              <ReferenceArea y1={80} y2={100} fill="#1b5e20" fillOpacity={0.05} />
              <ReferenceArea y1={60} y2={80} fill="#1565c0" fillOpacity={0.05} />
              <ReferenceArea y1={40} y2={60} fill="#ef6c00" fillOpacity={0.05} />
              <ReferenceArea y1={0} y2={40} fill="#c62828" fillOpacity={0.05} />
              <ReferenceLine y={80} stroke="#1b5e20" strokeDasharray="5 5" label={{ value: 'A', position: 'right', fontSize: 10 }} />
              <ReferenceLine y={60} stroke="#1565c0" strokeDasharray="5 5" label={{ value: 'B', position: 'right', fontSize: 10 }} />
              <ReferenceLine y={40} stroke="#ef6c00" strokeDasharray="5 5" label={{ value: 'C', position: 'right', fontSize: 10 }} />
              <XAxis dataKey="year" />
              <YAxis domain={[0, 100]} />
              <Tooltip
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Score']}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#1b5e20"
                strokeWidth={2.5}
                dot={{ r: 5, fill: '#1b5e20' }}
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
            No historical score data available.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default TrendChart;
