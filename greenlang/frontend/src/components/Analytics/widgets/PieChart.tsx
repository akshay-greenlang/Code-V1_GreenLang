/**
 * Pie/donut chart widget for categorical data visualization.
 */

import React, { useMemo } from 'react';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip
} from 'recharts';
import { Metric } from '../MetricService';

interface PieChartProps {
  title: string;
  data: Metric[];
  config?: {
    donut?: boolean;
    showPercentages?: boolean;
    colors?: string[];
  };
  onRemove?: () => void;
}

const DEFAULT_COLORS = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4'];

const PieChart: React.FC<PieChartProps> = ({ title, data, config, onRemove }) => {
  const chartData = useMemo(() => {
    // Group data by name and sum values
    const grouped: Record<string, number> = {};

    data.forEach(metric => {
      const name = metric.name || 'Unknown';
      const value = typeof metric.value === 'number' ? metric.value : 0;
      grouped[name] = (grouped[name] || 0) + value;
    });

    return Object.entries(grouped).map(([name, value]) => ({
      name,
      value
    }));
  }, [data]);

  const total = useMemo(() => {
    return chartData.reduce((sum, item) => sum + item.value, 0);
  }, [chartData]);

  const renderLabel = (entry: any) => {
    if (!config?.showPercentages) {
      return entry.name;
    }
    const percentage = total > 0 ? ((entry.value / total) * 100).toFixed(1) : 0;
    return `${entry.name}: ${percentage}%`;
  };

  const colors = config?.colors || DEFAULT_COLORS;

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
        <RechartsPieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderLabel}
            outerRadius={config?.donut ? "80%" : "70%"}
            innerRadius={config?.donut ? "50%" : 0}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => value.toFixed(2)} />
          <Legend />
        </RechartsPieChart>
      </ResponsiveContainer>

      {data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>No data available</div>
      )}
    </div>
  );
};

export default PieChart;
