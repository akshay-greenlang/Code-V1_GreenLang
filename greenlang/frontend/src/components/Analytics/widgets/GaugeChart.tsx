/**
 * Gauge chart widget for single metric visualization.
 */

import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Metric } from '../MetricService';

interface GaugeChartProps {
  title: string;
  data: Metric[];
  config?: {
    min?: number;
    max?: number;
    unit?: string;
    thresholds?: { value: number; color: string }[];
  };
  onRemove?: () => void;
}

const GaugeChart: React.FC<GaugeChartProps> = ({ title, data, config, onRemove }) => {
  const value = useMemo(() => {
    if (data.length === 0) return 0;
    const latest = data[data.length - 1];
    return typeof latest.value === 'number' ? latest.value : 0;
  }, [data]);

  const min = config?.min || 0;
  const max = config?.max || 100;
  const percentage = ((value - min) / (max - min)) * 100;

  const getColor = () => {
    if (!config?.thresholds) {
      return percentage < 70 ? '#4caf50' : percentage < 90 ? '#ff9800' : '#f44336';
    }

    for (const threshold of config.thresholds) {
      if (value <= threshold.value) {
        return threshold.color;
      }
    }
    return '#f44336';
  };

  const gaugeData = [
    { name: 'value', value: percentage },
    { name: 'empty', value: 100 - percentage }
  ];

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

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <ResponsiveContainer width="100%" height="70%">
          <PieChart>
            <Pie
              data={gaugeData}
              cx="50%"
              cy="70%"
              startAngle={180}
              endAngle={0}
              innerRadius="60%"
              outerRadius="90%"
              dataKey="value"
            >
              <Cell fill={getColor()} />
              <Cell fill="#e0e0e0" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div style={{ textAlign: 'center', marginTop: '8px' }}>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: getColor() }}>
            {value.toFixed(1)}{config?.unit || ''}
          </div>
          <div style={{ fontSize: '14px', color: '#999' }}>
            {min} - {max} {config?.unit || ''}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GaugeChart;
