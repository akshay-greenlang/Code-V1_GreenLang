/**
 * PathwayChart - Emissions reduction pathway visualization
 *
 * Plots the target emissions pathway from base year to target year
 * alongside actual emissions data, highlighting gaps between plan
 * and reality. Uses Recharts AreaChart with dual series.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import type { PathwayPoint } from '../../types';

interface PathwayChartProps {
  pathway: PathwayPoint[];
  baseYear: number;
  targetYear: number;
  baseYearEmissions: number;
  targetEmissions: number;
}

const PathwayChart: React.FC<PathwayChartProps> = ({
  pathway,
  baseYear,
  targetYear,
  baseYearEmissions,
  targetEmissions,
}) => {
  const reductionPct = baseYearEmissions > 0
    ? ((baseYearEmissions - targetEmissions) / baseYearEmissions * 100).toFixed(0)
    : '--';

  const currentYear = new Date().getFullYear();
  const latestActual = pathway
    .filter((p) => p.actual_emissions != null)
    .sort((a, b) => b.year - a.year)[0];

  const onTrack = latestActual
    ? (latestActual.actual_emissions! <= latestActual.target_emissions)
    : null;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">Emissions Reduction Pathway</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip label={`${baseYear} - ${targetYear}`} size="small" variant="outlined" />
            <Chip label={`-${reductionPct}%`} size="small" color="primary" />
            {onTrack !== null && (
              <Chip
                label={onTrack ? 'On Track' : 'Off Track'}
                size="small"
                color={onTrack ? 'success' : 'error'}
              />
            )}
          </Box>
        </Box>

        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={pathway}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="year"
              tick={{ fontSize: 11 }}
              domain={[baseYear, targetYear]}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v.toFixed(0)
              }
              label={{
                value: 'tCO2e',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 11 },
              }}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                `${value.toLocaleString()} tCO2e`,
                name === 'target_emissions' ? 'Target' : 'Actual',
              ]}
              labelFormatter={(label) => `Year: ${label}`}
            />
            <Legend />
            <ReferenceLine x={currentYear} stroke="#9e9e9e" strokeDasharray="4 4" label="Now" />
            <Area
              type="monotone"
              dataKey="target_emissions"
              name="Target Pathway"
              stroke="#1b5e20"
              fill="#c8e6c9"
              strokeWidth={2}
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="actual_emissions"
              name="Actual Emissions"
              stroke="#1565c0"
              fill="#bbdefb"
              strokeWidth={2}
              dot={{ r: 4 }}
              connectNulls={false}
            />
          </AreaChart>
        </ResponsiveContainer>

        <Box sx={{ display: 'flex', gap: 3, mt: 2, justifyContent: 'center' }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Base Year ({baseYear})</Typography>
            <Typography variant="body2" fontWeight={600}>
              {baseYearEmissions.toLocaleString()} tCO2e
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Target ({targetYear})</Typography>
            <Typography variant="body2" fontWeight={600}>
              {targetEmissions.toLocaleString()} tCO2e
            </Typography>
          </Box>
          {latestActual && (
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Latest Actual ({latestActual.year})
              </Typography>
              <Typography
                variant="body2"
                fontWeight={600}
                color={onTrack ? 'success.main' : 'error.main'}
              >
                {latestActual.actual_emissions!.toLocaleString()} tCO2e
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PathwayChart;
