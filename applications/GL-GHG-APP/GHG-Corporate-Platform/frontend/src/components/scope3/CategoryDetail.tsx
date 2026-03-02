/**
 * CategoryDetail - Detailed view of a single Scope 3 category
 *
 * Shows category name, description, calculation method, emissions
 * total with uncertainty range, top contributors (Pareto chart),
 * data sources, quality assessment, and year-over-year comparison.
 */

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  Alert,
} from '@mui/material';
import type { Scope3CategoryBreakdown } from '../../types';
import { SCOPE3_CATEGORY_NAMES } from '../../types';
import StatusBadge from '../common/StatusBadge';
import { formatNumber, formatPercent } from '../../utils/formatters';

interface CategoryDetailProps {
  category: Scope3CategoryBreakdown;
  previous?: Scope3CategoryBreakdown;
  onClose?: () => void;
}

const CategoryDetail: React.FC<CategoryDetailProps> = ({ category, previous }) => {
  const yoyChange = previous
    ? ((category.emissions_tco2e - previous.emissions_tco2e) / previous.emissions_tco2e) * 100
    : null;

  const contributors = category.top_contributors || [];

  // Compute cumulative percentage for Pareto
  const totalContrib = contributors.reduce((s, c) => s + c.emissions, 0);
  let cumulative = 0;
  const paretoData = contributors.map((c) => {
    cumulative += (c.emissions / (totalContrib || 1)) * 100;
    return {
      name: c.name.length > 20 ? c.name.slice(0, 20) + '...' : c.name,
      emissions: c.emissions,
      cumPercent: cumulative,
    };
  });

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <Chip
            label={`Category ${category.category_number}`}
            color="primary"
            variant="outlined"
          />
          <StatusBadge status={category.is_material ? 'material' : 'immaterial'} />
          <StatusBadge status={category.data_quality_tier} />
        </Box>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {category.category_name || SCOPE3_CATEGORY_NAMES[category.category]}
        </Typography>
      </Box>

      {/* Key metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">Total Emissions</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {formatNumber(category.emissions_tco2e)}
              </Typography>
              <Typography variant="body2" color="text.secondary">tCO2e</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">% of Scope 3</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {formatPercent(category.percentage_of_total)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">YoY Change</Typography>
              {yoyChange !== null ? (
                <Typography
                  variant="h4"
                  sx={{
                    fontWeight: 700,
                    color: yoyChange < 0 ? 'success.main' : yoyChange > 0 ? 'error.main' : 'text.primary',
                  }}
                >
                  {yoyChange > 0 ? '+' : ''}{yoyChange.toFixed(1)}%
                </Typography>
              ) : (
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.secondary' }}>
                  N/A
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Calculation method */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Calculation Method</Typography>
          <Chip
            label={category.calculation_method?.replace(/_/g, ' ') || 'Not specified'}
            variant="outlined"
          />
        </CardContent>
      </Card>

      {/* Top contributors Pareto chart */}
      {paretoData.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Top Contributors</Typography>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart
                data={paretoData}
                margin={{ top: 10, right: 30, left: 10, bottom: 50 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, angle: -35, textAnchor: 'end' }}
                  interval={0}
                />
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                  domain={[0, 100]}
                />
                <Tooltip
                  formatter={(value: number, name: string) =>
                    name === 'cumPercent'
                      ? [`${value.toFixed(1)}%`, 'Cumulative %']
                      : [`${formatNumber(value)} tCO2e`, 'Emissions']
                  }
                />
                <Bar
                  dataKey="emissions"
                  yAxisId="left"
                  fill="#43a047"
                  radius={[4, 4, 0, 0]}
                  barSize={28}
                />
                <Line
                  type="monotone"
                  dataKey="cumPercent"
                  yAxisId="right"
                  stroke="#ef6c00"
                  strokeWidth={2}
                  dot={{ fill: '#ef6c00', r: 3 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Exclusion reason if applicable */}
      {category.is_excluded && category.exclusion_reason && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Excluded:</strong> {category.exclusion_reason}
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default CategoryDetail;
