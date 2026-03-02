/**
 * DualReportingComparison - Location vs. Market-based side-by-side
 *
 * Shows two large stat cards comparing location-based and market-based
 * Scope 2 totals with a delta indicator, plus a grouped bar chart
 * showing both methods per facility.
 */

import React from 'react';
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
import { Box, Typography, Card, CardContent, Grid, Chip } from '@mui/material';
import { TrendingDown, TrendingUp } from '@mui/icons-material';
import type { Scope2Summary } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface DualReportingComparisonProps {
  summary: Scope2Summary;
}

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        p: 1.5,
        borderRadius: 1,
        boxShadow: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>{label}</Typography>
      {payload.map((entry) => (
        <Box key={entry.name} sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
          <Typography variant="body2" sx={{ color: entry.color }}>{entry.name}</Typography>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {formatNumber(entry.value, 1)} tCO2e
          </Typography>
        </Box>
      ))}
    </Box>
  );
};

const DualReportingComparison: React.FC<DualReportingComparisonProps> = ({ summary }) => {
  const delta = summary.location_based_tco2e - summary.market_based_tco2e;
  const deltaPercent = summary.location_based_tco2e > 0
    ? (delta / summary.location_based_tco2e) * 100
    : 0;
  const marketLower = delta > 0;

  const facilityData = summary.by_entity.map((entity) => ({
    name: entity.entity_name,
    locationBased: entity.emissions_tco2e,
    marketBased: entity.equity_share_emissions_tco2e,
  }));

  return (
    <Box>
      {/* Stat cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card sx={{ borderLeft: '4px solid #1e88e5' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Location-Based
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1e88e5' }}>
                {formatNumber(summary.location_based_tco2e, 1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">tCO2e</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ borderLeft: '4px solid #43a047' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Market-Based
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#43a047' }}>
                {formatNumber(summary.market_based_tco2e, 1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">tCO2e</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ borderLeft: `4px solid ${marketLower ? '#2e7d32' : '#c62828'}` }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Difference
              </Typography>
              <Typography
                variant="h4"
                sx={{ fontWeight: 700, color: marketLower ? '#2e7d32' : '#c62828' }}
              >
                {formatNumber(Math.abs(delta), 1)}
              </Typography>
              <Chip
                icon={marketLower ? <TrendingDown fontSize="small" /> : <TrendingUp fontSize="small" />}
                label={`${marketLower ? '-' : '+'}${Math.abs(deltaPercent).toFixed(1)}%`}
                size="small"
                color={marketLower ? 'success' : 'error'}
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Explanation */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            {marketLower
              ? 'Market-based emissions are lower than location-based, indicating the use of contractual instruments (RECs, PPAs, green tariffs) that reduce attributed emissions below the grid average.'
              : 'Market-based emissions exceed location-based, which may occur when contractual instruments have higher emission factors than the grid average, or when residual mix factors are applied.'}
          </Typography>
        </CardContent>
      </Card>

      {/* Grouped bar chart */}
      {facilityData.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Comparison by Entity
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={facilityData}
                margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, angle: -30, textAnchor: 'end' }}
                  interval={0}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="locationBased" name="Location-Based" fill="#1e88e5" radius={[4, 4, 0, 0]} barSize={20} />
                <Bar dataKey="marketBased" name="Market-Based" fill="#43a047" radius={[4, 4, 0, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default DualReportingComparison;
