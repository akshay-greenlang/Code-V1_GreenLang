/**
 * CategoryRadar - 17-category radar chart (Recharts)
 *
 * Displays a radar chart comparing scores across all 17
 * CDP scoring categories.
 */

import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, Tooltip,
} from 'recharts';
import type { CategoryScore } from '../../types';
import { SCORING_CATEGORY_NAMES, ScoringCategory } from '../../types';

interface CategoryRadarProps {
  categories: CategoryScore[];
}

const CategoryRadar: React.FC<CategoryRadarProps> = ({ categories }) => {
  const chartData = categories.map((cat) => ({
    category: SCORING_CATEGORY_NAMES[cat.category]?.split(' ').slice(0, 2).join(' ') || cat.category,
    fullName: SCORING_CATEGORY_NAMES[cat.category] || cat.category,
    score: cat.percentage,
    weight: cat.weight_management * 100,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Category Scores (17 Categories)
        </Typography>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid />
              <PolarAngleAxis dataKey="category" tick={{ fontSize: 9 }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(value: number, _name: string, props: { payload: { fullName: string } }) => [
                  `${value.toFixed(1)}%`,
                  props.payload.fullName,
                ]}
              />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#1b5e20"
                fill="#1b5e20"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
            No category data available.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default CategoryRadar;
