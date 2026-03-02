/**
 * ScopeDonut - Recharts PieChart showing Scope 1/2/3 breakdown
 *
 * Renders a donut chart with three color-coded segments representing
 * Scope 1 (red), Scope 2 (blue), and Scope 3 (green) emissions.
 * Displays the grand total in the center and percentage labels
 * on each segment with a legend showing scope names and tCO2e values.
 */

import React, { useCallback } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Label,
} from 'recharts';
import { Box, Typography, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { formatNumber } from '../../utils/formatters';
import { SCOPE_COLORS } from '../../types';

interface ScopeDonutProps {
  scope1: number;
  scope2Location: number;
  scope2Market: number;
  scope3: number;
  showMarketBased?: boolean;
  onToggleMethod?: (marketBased: boolean) => void;
}

interface DonutSegment {
  name: string;
  value: number;
  color: string;
  percentage: number;
}

const RADIAN = Math.PI / 180;

const renderCustomLabel = ({
  cx,
  cy,
  midAngle,
  innerRadius,
  outerRadius,
  percentage,
}: {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  percentage: number;
}) => {
  if (percentage < 3) return null;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="#ffffff"
      textAnchor="middle"
      dominantBaseline="central"
      fontSize={13}
      fontWeight={600}
    >
      {`${percentage.toFixed(1)}%`}
    </text>
  );
};

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: DonutSegment }> }) => {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
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
      <Typography variant="body2" sx={{ fontWeight: 600 }}>
        {item.name}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {formatNumber(item.value)} tCO2e ({item.percentage.toFixed(1)}%)
      </Typography>
    </Box>
  );
};

const renderLegend = (props: { payload?: Array<{ value: string; color: string; payload: DonutSegment }> }) => {
  const { payload } = props;
  if (!payload) return null;
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, mt: 1 }}>
      {payload.map((entry) => (
        <Box key={entry.value} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: entry.color,
              flexShrink: 0,
            }}
          />
          <Typography variant="body2" sx={{ flexGrow: 1 }}>
            {entry.value}
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {formatNumber(entry.payload.value)} tCO2e
          </Typography>
        </Box>
      ))}
    </Box>
  );
};

const ScopeDonut: React.FC<ScopeDonutProps> = ({
  scope1,
  scope2Location,
  scope2Market,
  scope3,
  showMarketBased = false,
  onToggleMethod,
}) => {
  const scope2Value = showMarketBased ? scope2Market : scope2Location;
  const total = scope1 + scope2Value + scope3;

  const data: DonutSegment[] = [
    {
      name: 'Scope 1 - Direct',
      value: scope1,
      color: SCOPE_COLORS.scope1,
      percentage: total > 0 ? (scope1 / total) * 100 : 0,
    },
    {
      name: `Scope 2 - ${showMarketBased ? 'Market' : 'Location'}`,
      value: scope2Value,
      color: SCOPE_COLORS.scope2,
      percentage: total > 0 ? (scope2Value / total) * 100 : 0,
    },
    {
      name: 'Scope 3 - Value Chain',
      value: scope3,
      color: SCOPE_COLORS.scope3,
      percentage: total > 0 ? (scope3 / total) * 100 : 0,
    },
  ];

  const handleToggle = useCallback(
    (_: React.MouseEvent<HTMLElement>, val: string | null) => {
      if (val !== null && onToggleMethod) {
        onToggleMethod(val === 'market');
      }
    },
    [onToggleMethod]
  );

  return (
    <Box>
      {onToggleMethod && (
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
          <ToggleButtonGroup
            value={showMarketBased ? 'market' : 'location'}
            exclusive
            onChange={handleToggle}
            size="small"
          >
            <ToggleButton value="location">Location</ToggleButton>
            <ToggleButton value="market">Market</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      )}
      <ResponsiveContainer width="100%" height={320}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={70}
            outerRadius={110}
            paddingAngle={2}
            dataKey="value"
            label={renderCustomLabel}
            labelLine={false}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
            ))}
            <Label
              value={`${formatNumber(total)}`}
              position="center"
              style={{ fontSize: 22, fontWeight: 700, fill: '#1a1a2e' }}
            />
            <Label
              value="tCO2e"
              position="center"
              dy={22}
              style={{ fontSize: 12, fill: '#4a4a68' }}
            />
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend content={renderLegend} />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ScopeDonut;
