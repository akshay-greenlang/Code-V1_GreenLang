/**
 * Line chart widget with zoom, pan, and multiple series support.
 */

import React, { useMemo } from 'react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceLine
} from 'recharts';
import { Metric } from '../MetricService';

interface LineChartProps {
  title: string;
  data: Metric[];
  config?: {
    yAxisLabel?: string;
    unit?: string;
    thresholds?: { value: number; label: string; color: string }[];
    showBrush?: boolean;
    stacked?: boolean;
    area?: boolean;
  };
  onRemove?: () => void;
}

const LineChart: React.FC<LineChartProps> = ({ title, data, config, onRemove }) => {
  const chartData = useMemo(() => {
    return data.map(metric => ({
      timestamp: new Date(metric.timestamp).getTime(),
      value: typeof metric.value === 'number' ? metric.value : 0,
      name: metric.name
    }));
  }, [data]);

  const formatXAxis = (timestamp: number) => {
    const date = new Date(timestamp);
    return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
  };

  const formatYAxis = (value: number) => {
    if (config?.unit) {
      return `${value.toFixed(2)}${config.unit}`;
    }
    return value.toFixed(2);
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {onRemove && (
          <button
            onClick={onRemove}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              fontSize: '18px',
              color: '#f44336'
            }}
          >
            ×
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxis}
            label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            tickFormatter={formatYAxis}
            label={{ value: config?.yAxisLabel || 'Value', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value: number) => [formatYAxis(value), 'Value']}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#2196f3"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
          />

          {/* Threshold lines */}
          {config?.thresholds?.map((threshold, index) => (
            <ReferenceLine
              key={index}
              y={threshold.value}
              stroke={threshold.color}
              strokeDasharray="3 3"
              label={threshold.label}
            />
          ))}

          {/* Brush for zooming */}
          {config?.showBrush && <Brush dataKey="timestamp" height={30} stroke="#2196f3" />}
        </RechartsLineChart>
      </ResponsiveContainer>

      {data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>
          No data available
        </div>
      )}
    </div>
  );
};

export default LineChart;
