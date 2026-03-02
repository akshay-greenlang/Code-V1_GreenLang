/**
 * GasBreakdown - Stacked bar chart of per-gas emissions
 *
 * Renders a Recharts stacked BarChart for the 7 GHG gases (CO2, CH4,
 * N2O, HFCs, PFCs, SF6, NF3). Supports toggling between absolute
 * (tCO2e) and percentage view, with optional previous year comparison.
 */

import React, { useState, useMemo } from 'react';
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
import { Box, Typography, Card, CardContent, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { GHGGas, GAS_COLORS } from '../../types';
import type { GHGGasBreakdown } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface GasBreakdownProps {
  gasBreakdown: GHGGasBreakdown;
  previousYear?: GHGGasBreakdown;
}

type ViewMode = 'absolute' | 'percentage';

const GAS_FIELDS: Array<{ key: keyof GHGGasBreakdown; gas: GHGGas; label: string }> = [
  { key: 'co2_tonnes', gas: GHGGas.CO2, label: 'CO2' },
  { key: 'ch4_tonnes_co2e', gas: GHGGas.CH4, label: 'CH4' },
  { key: 'n2o_tonnes_co2e', gas: GHGGas.N2O, label: 'N2O' },
  { key: 'hfcs_tonnes_co2e', gas: GHGGas.HFCs, label: 'HFCs' },
  { key: 'pfcs_tonnes_co2e', gas: GHGGas.PFCs, label: 'PFCs' },
  { key: 'sf6_tonnes_co2e', gas: GHGGas.SF6, label: 'SF6' },
  { key: 'nf3_tonnes_co2e', gas: GHGGas.NF3, label: 'NF3' },
];

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
        minWidth: 160,
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        {label}
      </Typography>
      {payload
        .filter((p) => p.value > 0)
        .map((entry) => (
          <Box key={entry.name} sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
            <Typography variant="body2" sx={{ color: entry.color }}>
              {entry.name}
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatNumber(entry.value, 1)}
            </Typography>
          </Box>
        ))}
    </Box>
  );
};

const GasBreakdown: React.FC<GasBreakdownProps> = ({ gasBreakdown, previousYear }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('absolute');

  const chartData = useMemo(() => {
    const makeRow = (label: string, breakdown: GHGGasBreakdown) => {
      const total = breakdown.total_co2e || 1;
      const row: Record<string, string | number> = { period: label };
      GAS_FIELDS.forEach(({ key, label: gasLabel }) => {
        const val = breakdown[key] as number;
        row[gasLabel] = viewMode === 'percentage' ? (val / total) * 100 : val;
      });
      return row;
    };

    const rows = [];
    if (previousYear) {
      rows.push(makeRow('Previous Year', previousYear));
    }
    rows.push(makeRow('Current Year', gasBreakdown));
    return rows;
  }, [gasBreakdown, previousYear, viewMode]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Gas Breakdown</Typography>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, v) => v && setViewMode(v)}
            size="small"
          >
            <ToggleButton value="absolute">tCO2e</ToggleButton>
            <ToggleButton value="percentage">%</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="period" tick={{ fontSize: 12 }} />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(v: number) =>
                viewMode === 'percentage'
                  ? `${v.toFixed(0)}%`
                  : v >= 1000
                    ? `${(v / 1000).toFixed(0)}K`
                    : String(v.toFixed(0))
              }
              label={{
                value: viewMode === 'percentage' ? '% of Total' : 'tCO2e',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 12 },
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {GAS_FIELDS.map(({ gas, label }) => (
              <Bar
                key={gas}
                dataKey={label}
                stackId="gases"
                fill={GAS_COLORS[gas]}
                name={label}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default GasBreakdown;
