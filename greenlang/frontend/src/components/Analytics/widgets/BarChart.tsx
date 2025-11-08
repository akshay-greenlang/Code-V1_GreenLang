/**
 * Bar chart widget with horizontal/vertical orientation and sorting.
 */

import React, { useMemo } from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Metric } from '../MetricService';

interface BarChartProps {
  title: string;
  data: Metric[];
  config?: {
    orientation?: 'horizontal' | 'vertical';
    stacked?: boolean;
    sorted?: boolean;
    colors?: string[];
  };
  onRemove?: () => void;
}

const DEFAULT_COLORS = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0'];

const BarChart: React.FC<BarChartProps> = ({ title, data, config, onRemove }) => {
  const chartData = useMemo(() => {
    let processedData = data.map((metric, index) => ({
      name: metric.name || `Item ${index}`,
      value: typeof metric.value === 'number' ? metric.value : 0
    }));

    if (config?.sorted) {
      processedData.sort((a, b) => b.value - a.value);
    }

    return processedData;
  }, [data, config?.sorted]);

  const isHorizontal = config?.orientation === 'horizontal';

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {onRemove && (
          <button onClick={onRemove} style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '18px', color: '#f44336' }}>
            ×
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <RechartsBarChart
          data={chartData}
          layout={isHorizontal ? 'horizontal' : 'vertical'}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          {isHorizontal ? (
            <>
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" />
            </>
          ) : (
            <>
              <XAxis dataKey="name" />
              <YAxis />
            </>
          )}
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#2196f3">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={(config?.colors || DEFAULT_COLORS)[index % (config?.colors || DEFAULT_COLORS).length]} />
            ))}
          </Bar>
        </RechartsBarChart>
      </ResponsiveContainer>

      {data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>No data available</div>
      )}
    </div>
  );
};

export default BarChart;
