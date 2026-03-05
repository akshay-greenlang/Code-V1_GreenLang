/**
 * ScopeComparisonChart - Side-by-side bar chart
 *
 * Renders a grouped bar chart comparing ISO 14064-1 categories
 * (6 categories) with GHG Protocol scopes (Scope 1/2/3).
 */

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { CrosswalkResult } from '../../types';
import { ISOCategory, ISO_CATEGORY_SHORT_NAMES, CATEGORY_COLORS } from '../../types';
import { formatNumber } from '../../utils/formatters';

const SCOPE_COLORS: Record<string, string> = {
  'Scope 1': '#e53935',
  'Scope 2': '#1e88e5',
  'Scope 3': '#43a047',
};

interface ScopeComparisonChartProps {
  crosswalk: CrosswalkResult;
}

const ScopeComparisonChart: React.FC<ScopeComparisonChartProps> = ({
  crosswalk,
}) => {
  // Aggregate by ISO category
  const isoCatData = useMemo(() => {
    const catTotals: Record<string, number> = {};
    crosswalk.mappings.forEach((m) => {
      const shortName = ISO_CATEGORY_SHORT_NAMES[m.iso_category as ISOCategory] || m.iso_category_name;
      catTotals[shortName] = (catTotals[shortName] || 0) + m.tco2e;
    });
    return Object.entries(catTotals).map(([name, value]) => ({
      name,
      value,
    }));
  }, [crosswalk]);

  // Aggregate by GHG scope
  const scopeData = useMemo(() => {
    const scopeTotals: Record<string, number> = {};
    crosswalk.mappings.forEach((m) => {
      scopeTotals[m.ghg_scope] = (scopeTotals[m.ghg_scope] || 0) + m.tco2e;
    });
    return Object.entries(scopeTotals).map(([name, value]) => ({
      name,
      value,
    }));
  }, [crosswalk]);

  // Combined chart data: side by side
  const chartData = useMemo(() => {
    const items = [
      ...isoCatData.map((d) => ({
        name: d.name,
        'ISO 14064-1': d.value,
        'GHG Protocol': 0,
        framework: 'ISO',
      })),
    ];

    // Add scope data as separate bars
    scopeData.forEach((d) => {
      items.push({
        name: d.name,
        'ISO 14064-1': 0,
        'GHG Protocol': d.value,
        framework: 'GHG',
      });
    });

    return items;
  }, [isoCatData, scopeData]);

  return (
    <Card>
      <CardHeader
        title="Framework Comparison"
        subheader="ISO 14064-1 Categories vs GHG Protocol Scopes"
      />
      <CardContent>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={chartData}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                height={70}
              />
              <YAxis
                label={{
                  value: 'tCO2e',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fontSize: 11 },
                }}
              />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${formatNumber(value, 2)} tCO2e`,
                  name,
                ]}
              />
              <Legend />
              <Bar
                dataKey="ISO 14064-1"
                fill="#1b5e20"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="GHG Protocol"
                fill="#1e88e5"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No crosswalk data available.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ScopeComparisonChart;
