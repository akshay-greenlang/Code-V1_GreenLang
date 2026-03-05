/**
 * ImpactProjectionChart - Stacked bar chart of planned reductions
 *
 * Renders a stacked bar chart showing planned emission reductions
 * by action category (emission reduction, removal enhancement,
 * data improvement, process improvement).
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
import type { ManagementAction } from '../../types';
import { ActionCategory } from '../../types';
import { formatNumber } from '../../utils/formatters';

const CATEGORY_COLORS: Record<ActionCategory, string> = {
  [ActionCategory.EMISSION_REDUCTION]: '#e53935',
  [ActionCategory.REMOVAL_ENHANCEMENT]: '#1b5e20',
  [ActionCategory.DATA_IMPROVEMENT]: '#1e88e5',
  [ActionCategory.PROCESS_IMPROVEMENT]: '#ef6c00',
};

const CATEGORY_LABELS: Record<ActionCategory, string> = {
  [ActionCategory.EMISSION_REDUCTION]: 'Emission Reduction',
  [ActionCategory.REMOVAL_ENHANCEMENT]: 'Removal Enhancement',
  [ActionCategory.DATA_IMPROVEMENT]: 'Data Improvement',
  [ActionCategory.PROCESS_IMPROVEMENT]: 'Process Improvement',
};

interface ImpactProjectionChartProps {
  actions: ManagementAction[];
}

const ImpactProjectionChart: React.FC<ImpactProjectionChartProps> = ({
  actions,
}) => {
  const chartData = useMemo(() => {
    // Group actions by status for stacked bars
    const statusGroups: Record<string, Record<string, number>> = {};

    actions.forEach((action) => {
      const status = action.status;
      const cat = action.action_category;
      const val = action.estimated_reduction_tco2e ?? 0;
      if (!statusGroups[status]) {
        statusGroups[status] = {};
      }
      statusGroups[status][cat] = (statusGroups[status][cat] || 0) + val;
    });

    return Object.entries(statusGroups).map(([status, catValues]) => ({
      status: status.replace(/_/g, ' '),
      [ActionCategory.EMISSION_REDUCTION]:
        catValues[ActionCategory.EMISSION_REDUCTION] || 0,
      [ActionCategory.REMOVAL_ENHANCEMENT]:
        catValues[ActionCategory.REMOVAL_ENHANCEMENT] || 0,
      [ActionCategory.DATA_IMPROVEMENT]:
        catValues[ActionCategory.DATA_IMPROVEMENT] || 0,
      [ActionCategory.PROCESS_IMPROVEMENT]:
        catValues[ActionCategory.PROCESS_IMPROVEMENT] || 0,
    }));
  }, [actions]);

  const totalReduction = actions.reduce(
    (sum, a) => sum + (a.estimated_reduction_tco2e ?? 0),
    0,
  );
  const totalCost = actions.reduce(
    (sum, a) => sum + (a.estimated_cost_usd ?? 0),
    0,
  );

  return (
    <Card>
      <CardHeader
        title="Impact Projection"
        subheader={`Total planned reduction: ${formatNumber(totalReduction, 0)} tCO2e | Investment: $${formatNumber(totalCost, 0)}`}
      />
      <CardContent>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="status" tick={{ fontSize: 12 }} />
              <YAxis
                label={{
                  value: 'Planned Reduction (tCO2e)',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fontSize: 11 },
                }}
              />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${formatNumber(value, 1)} tCO2e`,
                  CATEGORY_LABELS[name as ActionCategory] || name,
                ]}
              />
              <Legend
                formatter={(value: string) =>
                  CATEGORY_LABELS[value as ActionCategory] || value
                }
              />
              {Object.values(ActionCategory).map((cat) => (
                <Bar
                  key={cat}
                  dataKey={cat}
                  stackId="a"
                  fill={CATEGORY_COLORS[cat]}
                  radius={[2, 2, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No management actions defined yet. Add actions to see impact projections.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ImpactProjectionChart;
