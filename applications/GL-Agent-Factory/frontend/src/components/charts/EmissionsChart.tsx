/**
 * EmissionsChart Component
 *
 * Interactive emissions data visualization using Recharts.
 */

import * as React from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { SkeletonChart } from '@/components/ui/Skeleton';
import { formatEmissions, formatNumber } from '@/utils/format';

// Color palette for charts
const COLORS = {
  primary: '#16a34a',
  secondary: '#4ade80',
  tertiary: '#86efac',
  quaternary: '#dcfce7',
  destructive: '#ef4444',
  warning: '#f59e0b',
  muted: '#94a3b8',
};

const PIE_COLORS = [
  '#16a34a',
  '#2563eb',
  '#7c3aed',
  '#db2777',
  '#f59e0b',
  '#06b6d4',
  '#84cc16',
  '#f97316',
];

// Custom tooltip
interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
    dataKey: string;
  }>;
  label?: string;
  formatter?: (value: number) => string;
}

const CustomTooltip: React.FC<TooltipProps> = ({ active, payload, label, formatter }) => {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-lg">
      <p className="font-medium mb-2">{label}</p>
      {payload.map((entry, index) => (
        <div key={index} className="flex items-center gap-2 text-sm">
          <span
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">
            {formatter ? formatter(entry.value) : formatNumber(entry.value)}
          </span>
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// Emissions Trend Chart
// ============================================================================

interface EmissionsTrendData {
  date: string;
  emissions: number;
  target?: number;
}

interface EmissionsTrendChartProps {
  data: EmissionsTrendData[];
  title?: string;
  description?: string;
  loading?: boolean;
  height?: number;
  showTarget?: boolean;
}

export function EmissionsTrendChart({
  data,
  title = 'Emissions Trend',
  description,
  loading,
  height = 300,
  showTarget = false,
}: EmissionsTrendChartProps) {
  const [chartType, setChartType] = React.useState<'line' | 'area'>('area');

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <SkeletonChart />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </div>
        <Tabs value={chartType} onValueChange={(v) => setChartType(v as 'line' | 'area')}>
          <TabsList>
            <TabsTrigger value="area">Area</TabsTrigger>
            <TabsTrigger value="line">Line</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          {chartType === 'area' ? (
            <AreaChart data={data}>
              <defs>
                <linearGradient id="emissionsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} stroke="#94a3b8" />
              <YAxis
                tick={{ fontSize: 12 }}
                stroke="#94a3b8"
                tickFormatter={(value) => formatNumber(value / 1000) + 'k'}
              />
              <Tooltip content={<CustomTooltip formatter={(v) => formatEmissions(v)} />} />
              <Legend />
              <Area
                type="monotone"
                dataKey="emissions"
                name="Emissions"
                stroke={COLORS.primary}
                fill="url(#emissionsGradient)"
                strokeWidth={2}
              />
              {showTarget && (
                <Line
                  type="monotone"
                  dataKey="target"
                  name="Target"
                  stroke={COLORS.destructive}
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={false}
                />
              )}
            </AreaChart>
          ) : (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} stroke="#94a3b8" />
              <YAxis
                tick={{ fontSize: 12 }}
                stroke="#94a3b8"
                tickFormatter={(value) => formatNumber(value / 1000) + 'k'}
              />
              <Tooltip content={<CustomTooltip formatter={(v) => formatEmissions(v)} />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="emissions"
                name="Emissions"
                stroke={COLORS.primary}
                strokeWidth={2}
                dot={{ fill: COLORS.primary, strokeWidth: 2 }}
              />
              {showTarget && (
                <Line
                  type="monotone"
                  dataKey="target"
                  name="Target"
                  stroke={COLORS.destructive}
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={false}
                />
              )}
            </LineChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Emissions By Category Chart (Pie/Donut)
// ============================================================================

interface EmissionsByCategoryData {
  category: string;
  value: number;
  percentage: number;
}

interface EmissionsByCategoryChartProps {
  data: EmissionsByCategoryData[];
  title?: string;
  description?: string;
  loading?: boolean;
  height?: number;
}

export function EmissionsByCategoryChart({
  data,
  title = 'Emissions by Category',
  description,
  loading,
  height = 300,
}: EmissionsByCategoryChartProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <SkeletonChart />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
              nameKey="category"
              label={({ category, percentage }) => `${category}: ${percentage.toFixed(1)}%`}
              labelLine={true}
            >
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const item = payload[0].payload as EmissionsByCategoryData;
                return (
                  <div className="rounded-lg border bg-background p-3 shadow-lg">
                    <p className="font-medium">{item.category}</p>
                    <p className="text-sm text-muted-foreground">
                      {formatEmissions(item.value)} ({item.percentage.toFixed(1)}%)
                    </p>
                  </div>
                );
              }}
            />
          </PieChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap justify-center gap-4">
          {data.map((item, index) => (
            <div key={item.category} className="flex items-center gap-2">
              <span
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: PIE_COLORS[index % PIE_COLORS.length] }}
              />
              <span className="text-sm text-muted-foreground">{item.category}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Emissions Comparison Bar Chart
// ============================================================================

interface EmissionsComparisonData {
  name: string;
  current: number;
  previous?: number;
  target?: number;
}

interface EmissionsComparisonChartProps {
  data: EmissionsComparisonData[];
  title?: string;
  description?: string;
  loading?: boolean;
  height?: number;
}

export function EmissionsComparisonChart({
  data,
  title = 'Emissions Comparison',
  description,
  loading,
  height = 300,
}: EmissionsComparisonChartProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <SkeletonChart />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal />
            <XAxis
              type="number"
              tick={{ fontSize: 12 }}
              stroke="#94a3b8"
              tickFormatter={(value) => formatNumber(value / 1000) + 'k'}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 12 }}
              stroke="#94a3b8"
              width={100}
            />
            <Tooltip content={<CustomTooltip formatter={(v) => formatEmissions(v)} />} />
            <Legend />
            <Bar dataKey="current" name="Current" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
            <Bar dataKey="previous" name="Previous" fill={COLORS.muted} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
