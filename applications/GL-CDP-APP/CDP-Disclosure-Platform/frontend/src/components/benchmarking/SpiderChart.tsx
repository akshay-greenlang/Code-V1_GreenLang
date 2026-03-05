/**
 * SpiderChart - Category comparison radar between org and sector avg
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { CategoryScore, ScoringCategory } from '../../types';
import { SCORING_CATEGORY_NAMES } from '../../types';

interface SpiderChartProps { orgScores: CategoryScore[]; sectorAverages: Record<ScoringCategory, number>; }

const SpiderChart: React.FC<SpiderChartProps> = ({ orgScores, sectorAverages }) => {
  const data = orgScores.map((s) => ({
    category: (SCORING_CATEGORY_NAMES[s.category] || s.category).split(' ').slice(0, 2).join(' '),
    yours: s.percentage,
    sector: (sectorAverages[s.category] || 0) * 100,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Category Comparison vs Sector</Typography>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={data} cx="50%" cy="50%" outerRadius="65%">
            <PolarGrid />
            <PolarAngleAxis dataKey="category" tick={{ fontSize: 9 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip />
            <Legend />
            <Radar name="Your Score" dataKey="yours" stroke="#1b5e20" fill="#1b5e20" fillOpacity={0.2} strokeWidth={2} />
            <Radar name="Sector Average" dataKey="sector" stroke="#ef6c00" fill="#ef6c00" fillOpacity={0.1} strokeWidth={2} strokeDasharray="5 5" />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default SpiderChart;
