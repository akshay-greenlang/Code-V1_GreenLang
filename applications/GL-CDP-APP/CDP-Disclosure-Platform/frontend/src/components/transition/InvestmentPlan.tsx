/**
 * InvestmentPlan - Transition investment breakdown
 *
 * Shows total investment, low-carbon revenue share, and
 * investment by technology lever in a donut chart.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import { AttachMoney, TrendingUp } from '@mui/icons-material';
import type { TechLever } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface InvestmentPlanProps {
  levers: TechLever[];
  totalInvestment: number;
  lowCarbonRevenuePct: number;
}

const PIE_COLORS = [
  '#1b5e20', '#2e7d32', '#1565c0', '#1e88e5',
  '#7b1fa2', '#00838f', '#ef6c00', '#e53935',
];

const InvestmentPlan: React.FC<InvestmentPlanProps> = ({
  levers,
  totalInvestment,
  lowCarbonRevenuePct,
}) => {
  const sorted = [...levers].sort(
    (a, b) => b.investment_required_usd - a.investment_required_usd,
  );

  const pieData = sorted.map((lever) => ({
    name: lever.name,
    value: lever.investment_required_usd,
  }));

  const leverTotal = levers.reduce((s, l) => s + l.investment_required_usd, 0);
  const costPerTonne = levers.reduce((s, l) => s + l.reduction_potential_tco2e, 0);
  const avgCostPerTonne = costPerTonne > 0
    ? leverTotal / costPerTonne
    : 0;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Investment Plan
        </Typography>

        {/* Top-level KPIs */}
        <Box sx={{ display: 'flex', gap: 3, mb: 3 }}>
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <AttachMoney sx={{ color: '#1b5e20', fontSize: 28 }} />
            <Typography variant="h5" fontWeight={700}>
              {formatCurrency(totalInvestment)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Total Investment
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <TrendingUp sx={{ color: '#1565c0', fontSize: 28 }} />
            <Typography variant="h5" fontWeight={700}>
              {lowCarbonRevenuePct.toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Low-Carbon Revenue
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <AttachMoney sx={{ color: '#ef6c00', fontSize: 28 }} />
            <Typography variant="h5" fontWeight={700}>
              {formatCurrency(avgCostPerTonne)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Avg Cost / tCO2e
            </Typography>
          </Box>
        </Box>

        {/* Investment donut */}
        {pieData.length > 0 && (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
              >
                {pieData.map((_entry, index) => (
                  <Cell
                    key={index}
                    fill={PIE_COLORS[index % PIE_COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => [formatCurrency(value), 'Investment']}
              />
              <Legend
                formatter={(value: string) =>
                  value.length > 25 ? value.slice(0, 23) + '...' : value
                }
              />
            </PieChart>
          </ResponsiveContainer>
        )}

        {/* Investment breakdown */}
        <Box sx={{ mt: 1 }}>
          {sorted.map((lever, idx) => {
            const pct = totalInvestment > 0
              ? (lever.investment_required_usd / totalInvestment * 100)
              : 0;
            return (
              <Box
                key={lever.id}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  py: 0.5,
                  borderBottom: idx < sorted.length - 1 ? '1px solid #f5f5f5' : 'none',
                }}
              >
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: PIE_COLORS[idx % PIE_COLORS.length],
                    flexShrink: 0,
                  }}
                />
                <Typography variant="body2" sx={{ flex: 1 }}>
                  {lever.name}
                </Typography>
                <Chip
                  label={`${pct.toFixed(0)}%`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: 10, minWidth: 40 }}
                />
                <Typography variant="body2" fontWeight={500} sx={{ minWidth: 80, textAlign: 'right' }}>
                  {formatCurrency(lever.investment_required_usd)}
                </Typography>
              </Box>
            );
          })}
        </Box>
      </CardContent>
    </Card>
  );
};

export default InvestmentPlan;
