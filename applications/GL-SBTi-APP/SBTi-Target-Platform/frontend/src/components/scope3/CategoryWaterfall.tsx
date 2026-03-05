/**
 * CategoryWaterfall - 15-category waterfall chart.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { CategoryBreakdown } from '../../types';

interface CategoryWaterfallProps { categories: CategoryBreakdown[]; }

const CategoryWaterfall: React.FC<CategoryWaterfallProps> = ({ categories }) => {
  const data = categories.sort((a, b) => b.emissions_tco2e - a.emissions_tco2e).map((c) => ({
    name: `Cat ${c.category_number}`,
    fullName: c.category_name,
    emissions: c.emissions_tco2e,
    pct: c.percentage_of_scope3,
    included: c.included_in_target,
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Scope 3 Category Breakdown</Typography>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={10} angle={-45} textAnchor="end" height={60} />
            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} fontSize={11} />
            <Tooltip formatter={(value: number, name: string, props: any) => [`${value.toLocaleString()} tCO2e (${props.payload.pct.toFixed(1)}%)`, props.payload.fullName]} />
            <Bar dataKey="emissions" name="Emissions">
              {data.map((d, i) => <Cell key={i} fill={d.included ? '#1B5E20' : '#BDBDBD'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default CategoryWaterfall;
