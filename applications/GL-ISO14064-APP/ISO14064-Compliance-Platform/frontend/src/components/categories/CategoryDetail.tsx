/**
 * CategoryDetail - Single category detail view
 *
 * Shows the emission sources table, gas breakdown pie chart (Recharts),
 * and facility breakdown bar chart for a single ISO 14064-1 category.
 */

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip as RTooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as BarTooltip,
  ResponsiveContainer,
} from 'recharts';
import type { CategoryResult } from '../../types';
import {
  ISO_CATEGORY_NAMES,
  CATEGORY_COLORS,
  GAS_COLORS,
  GHGGas,
} from '../../types';
import { formatTCO2e, formatNumber, getStatusColor, getDataQualityLabel, getDataQualityColor } from '../../utils/formatters';

interface CategoryDetailProps {
  category: CategoryResult;
}

const CategoryDetail: React.FC<CategoryDetailProps> = ({ category }) => {
  const gasData = useMemo(() => {
    return Object.entries(category.by_gas)
      .filter(([, v]) => v > 0)
      .map(([gas, value]) => ({
        name: gas,
        value,
        color: GAS_COLORS[gas as GHGGas] || '#9e9e9e',
      }));
  }, [category.by_gas]);

  const facilityData = useMemo(() => {
    return Object.entries(category.by_facility)
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([name, value]) => ({
        name: name.length > 20 ? name.substring(0, 17) + '...' : name,
        value,
      }));
  }, [category.by_facility]);

  const color = CATEGORY_COLORS[category.category];

  return (
    <Card sx={{ borderTop: `3px solid ${color}` }}>
      <CardHeader
        title={ISO_CATEGORY_NAMES[category.category]}
        subheader={
          <Box sx={{ display: 'flex', gap: 1, mt: 0.5, flexWrap: 'wrap' }}>
            <Chip
              label={category.significance.replace(/_/g, ' ')}
              color={getStatusColor(category.significance)}
              size="small"
            />
            <Chip
              label={getDataQualityLabel(category.data_quality_tier)}
              color={getDataQualityColor(category.data_quality_tier)}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`${category.source_count} sources`}
              size="small"
              variant="outlined"
            />
          </Box>
        }
      />
      <CardContent>
        {/* Summary metrics */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Gross Emissions
              </Typography>
              <Typography variant="h6" fontWeight={700} sx={{ color }}>
                {formatTCO2e(category.total_tco2e)}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Removals
              </Typography>
              <Typography variant="h6" fontWeight={700} color="success.main">
                {formatTCO2e(category.removals_tco2e)}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Net Emissions
              </Typography>
              <Typography variant="h6" fontWeight={700}>
                {formatTCO2e(category.net_tco2e)}
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Charts */}
        <Grid container spacing={3}>
          {/* Gas breakdown pie */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Gas Breakdown
            </Typography>
            {gasData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={gasData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={100}
                    paddingAngle={2}
                  >
                    {gasData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Pie>
                  <RTooltip
                    formatter={(value: number) => formatNumber(value, 2) + ' tCO2e'}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                No gas breakdown data available.
              </Typography>
            )}
          </Grid>

          {/* Facility breakdown bar */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Facility Breakdown
            </Typography>
            {facilityData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={facilityData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
                  <BarTooltip
                    formatter={(value: number) => formatNumber(value, 2) + ' tCO2e'}
                  />
                  <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                No facility breakdown data available.
              </Typography>
            )}
          </Grid>
        </Grid>

        {/* Biogenic CO2 */}
        {category.biogenic_co2 > 0 && (
          <Box sx={{ mt: 2, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Biogenic CO2 (reported separately per ISO 14064-1): {formatTCO2e(category.biogenic_co2)}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default CategoryDetail;
