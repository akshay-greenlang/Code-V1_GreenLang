/**
 * ReconciliationWaterfall - Location-to-market bridge chart
 *
 * Recharts waterfall chart showing the bridge from location-based
 * to market-based Scope 2 total. Starting bar is location-based,
 * adjustment bars show RECs, PPAs, green tariffs, and residual mix
 * impacts, ending with the market-based total.
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Box, Typography, Card, CardContent } from '@mui/material';
import type { ReconciliationData } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface ReconciliationWaterfallProps {
  reconciliation: ReconciliationData;
}

interface WaterfallEntry {
  name: string;
  value: number;
  base: number;
  barValue: number;
  isTotal: boolean;
  color: string;
}

const ReconciliationWaterfall: React.FC<ReconciliationWaterfallProps> = ({ reconciliation }) => {
  const data: WaterfallEntry[] = useMemo(() => {
    const entries: WaterfallEntry[] = [];
    let running = reconciliation.location_total;

    // Start: location-based total
    entries.push({
      name: 'Location-Based',
      value: reconciliation.location_total,
      base: 0,
      barValue: reconciliation.location_total,
      isTotal: true,
      color: '#1e88e5',
    });

    // Adjustments
    reconciliation.adjustments.forEach((adj) => {
      const adjValue = adj.type === 'reduction' ? -Math.abs(adj.value) : Math.abs(adj.value);
      const newRunning = running + adjValue;
      entries.push({
        name: adj.name,
        value: adjValue,
        base: adjValue < 0 ? newRunning : running,
        barValue: Math.abs(adjValue),
        isTotal: false,
        color: adjValue < 0 ? '#2e7d32' : '#c62828',
      });
      running = newRunning;
    });

    // End: market-based total
    entries.push({
      name: 'Market-Based',
      value: reconciliation.market_total,
      base: 0,
      barValue: reconciliation.market_total,
      isTotal: true,
      color: '#43a047',
    });

    return entries;
  }, [reconciliation]);

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ payload: WaterfallEntry }>;
  }) => {
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
        <Typography variant="subtitle2">{item.name}</Typography>
        <Typography variant="body2">
          {item.isTotal
            ? `${formatNumber(item.value)} tCO2e`
            : `${item.value > 0 ? '+' : ''}${formatNumber(item.value)} tCO2e`}
        </Typography>
      </Box>
    );
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Location to Market Reconciliation
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Waterfall showing adjustments that bridge location-based to market-based Scope 2 total.
        </Typography>

        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={data}
            margin={{ top: 20, right: 20, left: 10, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11, angle: -25, textAnchor: 'end' }}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v))}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Invisible base bar for stacking */}
            <Bar dataKey="base" stackId="waterfall" fill="transparent" />

            {/* Visible bar */}
            <Bar dataKey="barValue" stackId="waterfall" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default ReconciliationWaterfall;
