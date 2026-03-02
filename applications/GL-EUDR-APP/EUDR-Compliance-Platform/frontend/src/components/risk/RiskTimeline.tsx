/**
 * RiskTimeline - Recharts line chart showing risk score over time.
 *
 * X-axis: months, Y-axis: risk score (0-1). Colored reference zones,
 * threshold lines at high (0.7) and critical (0.9), and tooltips with
 * date, score, and event count.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';
import type { RiskTrendPoint } from '../../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface RiskTimelineProps {
  trends: RiskTrendPoint[];
  thresholds?: {
    high: number;
    critical: number;
  };
  height?: number;
  title?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatMonth(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' });
}

function riskLevelColor(score: number): string {
  if (score >= 0.9) return '#b71c1c';
  if (score >= 0.7) return '#f44336';
  if (score >= 0.5) return '#ff9800';
  if (score >= 0.3) return '#fdd835';
  return '#4caf50';
}

// ---------------------------------------------------------------------------
// Custom Tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload as RiskTrendPoint;
  const score = data.risk_score;
  const color = riskLevelColor(score);

  return (
    <Box
      sx={{
        backgroundColor: 'white',
        border: '1px solid #e0e0e0',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
        minWidth: 160,
      }}
    >
      <Typography variant="subtitle2" gutterBottom>
        {new Date(data.date).toLocaleDateString('en-GB', {
          day: 'numeric',
          month: 'long',
          year: 'numeric',
        })}
      </Typography>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Box
          sx={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            backgroundColor: color,
          }}
        />
        <Typography variant="body2">
          Score: <strong>{(score * 100).toFixed(1)}%</strong>
        </Typography>
      </Stack>
      <Typography variant="body2" sx={{ mt: 0.5 }}>
        Risk Level:{' '}
        <strong style={{ textTransform: 'capitalize', color }}>
          {data.risk_level.replace('_', ' ')}
        </strong>
      </Typography>
      {data.event_count > 0 && (
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          Events: <strong>{data.event_count}</strong>
        </Typography>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const RiskTimeline: React.FC<RiskTimelineProps> = ({
  trends,
  thresholds = { high: 0.7, critical: 0.9 },
  height = 300,
  title = 'Risk Score Trend',
}) => {
  const chartData = useMemo(
    () =>
      trends.map((t) => ({
        ...t,
        monthLabel: formatMonth(t.date),
      })),
    [trends]
  );

  if (trends.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="subtitle1">{title}</Typography>
          <Typography color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
            No risk trend data available.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          {title}
        </Typography>

        <ResponsiveContainer width="100%" height={height}>
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 20, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />

            {/* Background zones */}
            <ReferenceArea y1={0} y2={0.3} fill="#e8f5e9" fillOpacity={0.5} />
            <ReferenceArea y1={0.3} y2={0.5} fill="#fffde7" fillOpacity={0.5} />
            <ReferenceArea y1={0.5} y2={thresholds.high} fill="#fff3e0" fillOpacity={0.5} />
            <ReferenceArea y1={thresholds.high} y2={thresholds.critical} fill="#fbe9e7" fillOpacity={0.5} />
            <ReferenceArea y1={thresholds.critical} y2={1} fill="#ffebee" fillOpacity={0.5} />

            <XAxis
              dataKey="monthLabel"
              tick={{ fontSize: 12, fill: '#666' }}
              tickLine={false}
              axisLine={{ stroke: '#e0e0e0' }}
            />

            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 12, fill: '#666' }}
              tickLine={false}
              axisLine={{ stroke: '#e0e0e0' }}
              width={45}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Threshold lines */}
            <ReferenceLine
              y={thresholds.high}
              stroke="#ff9800"
              strokeDasharray="6 3"
              label={{
                value: 'High',
                position: 'right',
                fill: '#ff9800',
                fontSize: 11,
              }}
            />
            <ReferenceLine
              y={thresholds.critical}
              stroke="#f44336"
              strokeDasharray="6 3"
              label={{
                value: 'Critical',
                position: 'right',
                fill: '#f44336',
                fontSize: 11,
              }}
            />

            {/* Risk score line */}
            <Line
              type="monotone"
              dataKey="risk_score"
              stroke="#1565c0"
              strokeWidth={2.5}
              dot={(props: Record<string, unknown>) => {
                const { cx, cy, payload } = props as { cx: number; cy: number; payload: RiskTrendPoint };
                const color = riskLevelColor(payload.risk_score);
                return (
                  <circle
                    key={`dot-${payload.date}`}
                    cx={cx}
                    cy={cy}
                    r={payload.event_count > 0 ? 6 : 4}
                    fill={color}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                );
              }}
              activeDot={{ r: 7, stroke: '#1565c0', strokeWidth: 2, fill: '#fff' }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Legend */}
        <Stack
          direction="row"
          spacing={2}
          justifyContent="center"
          mt={1}
          flexWrap="wrap"
          useFlexGap
        >
          {[
            { label: 'Low (0-30%)', color: '#c8e6c9' },
            { label: 'Standard (30-50%)', color: '#fff9c4' },
            { label: 'High (50-70%)', color: '#ffe0b2' },
            { label: 'Critical (70-100%)', color: '#ffcdd2' },
          ].map((item) => (
            <Stack key={item.label} direction="row" alignItems="center" spacing={0.5}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '2px',
                  backgroundColor: item.color,
                  border: '1px solid rgba(0,0,0,0.1)',
                }}
              />
              <Typography variant="caption" color="text.secondary">
                {item.label}
              </Typography>
            </Stack>
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default RiskTimeline;
