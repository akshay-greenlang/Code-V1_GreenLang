/**
 * TechLeverBreakdown - Technology lever contribution analysis
 *
 * Visualizes the emissions reduction potential and investment
 * required for each technology lever in the transition plan.
 * Uses a horizontal bar chart and summary table.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { TechLever } from '../../types';
import { formatCurrency, formatTCO2e } from '../../utils/formatters';

interface TechLeverBreakdownProps {
  levers: TechLever[];
  totalReduction: number;
}

const LEVER_COLORS = [
  '#1b5e20', '#2e7d32', '#1565c0', '#1e88e5',
  '#7b1fa2', '#00838f', '#ef6c00', '#e53935',
  '#6d4c41', '#455a64',
];

const TechLeverBreakdown: React.FC<TechLeverBreakdownProps> = ({
  levers,
  totalReduction,
}) => {
  const sorted = [...levers].sort(
    (a, b) => b.reduction_potential_tco2e - a.reduction_potential_tco2e,
  );

  const chartData = sorted.map((lever) => ({
    name: lever.name.length > 20 ? lever.name.slice(0, 18) + '...' : lever.name,
    fullName: lever.name,
    reduction: lever.reduction_potential_tco2e,
    investment: lever.investment_required_usd,
    pct: totalReduction > 0
      ? (lever.reduction_potential_tco2e / totalReduction * 100)
      : 0,
  }));

  const totalInvestment = levers.reduce(
    (sum, l) => sum + l.investment_required_usd, 0,
  );

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">Technology Levers</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`${levers.length} levers`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`Total: ${formatCurrency(totalInvestment)}`}
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
        </Box>

        <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 40 + 40)}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v.toFixed(0)
              }
              label={{
                value: 'Reduction (tCO2e)',
                position: 'insideBottom',
                offset: -5,
                style: { fontSize: 11 },
              }}
            />
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fontSize: 11 }}
              width={150}
            />
            <Tooltip
              formatter={(value: number, _name: string, entry: { payload: { fullName: string; pct: number; investment: number } }) => {
                const { fullName, pct, investment } = entry.payload;
                return [
                  `${value.toLocaleString()} tCO2e (${pct.toFixed(1)}%) | Investment: ${formatCurrency(investment)}`,
                  fullName,
                ];
              }}
            />
            <Bar dataKey="reduction" radius={[0, 4, 4, 0]}>
              {chartData.map((_entry, index) => (
                <Cell key={index} fill={LEVER_COLORS[index % LEVER_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Summary list */}
        <Box sx={{ mt: 2 }}>
          {sorted.map((lever, idx) => (
            <Box
              key={lever.id}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                py: 0.75,
                borderBottom: idx < sorted.length - 1 ? '1px solid #f0f0f0' : 'none',
              }}
            >
              <Box
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  bgcolor: LEVER_COLORS[idx % LEVER_COLORS.length],
                  flexShrink: 0,
                }}
              />
              <Typography variant="body2" sx={{ flex: 1, fontWeight: 500 }}>
                {lever.name}
              </Typography>
              <Chip
                label={lever.maturity}
                size="small"
                variant="outlined"
                sx={{ fontSize: 10 }}
              />
              <Chip
                label={lever.timeline}
                size="small"
                variant="outlined"
                sx={{ fontSize: 10 }}
              />
              <Typography variant="body2" sx={{ minWidth: 100, textAlign: 'right' }}>
                {formatTCO2e(lever.reduction_potential_tco2e)}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ minWidth: 90, textAlign: 'right' }}
              >
                {formatCurrency(lever.investment_required_usd)}
              </Typography>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default TechLeverBreakdown;
