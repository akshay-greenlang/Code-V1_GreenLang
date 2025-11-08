/**
 * Statistic card widget with trend indicator and sparkline.
 */

import React, { useMemo } from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { Metric } from '../MetricService';

interface StatCardProps {
  title: string;
  data: Metric[];
  config?: {
    format?: 'number' | 'bytes' | 'duration' | 'percentage';
    decimals?: number;
    showSparkline?: boolean;
    showTrend?: boolean;
  };
  onRemove?: () => void;
}

const StatCard: React.FC<StatCardProps> = ({ title, data, config, onRemove }) => {
  const { currentValue, previousValue, trend, sparklineData } = useMemo(() => {
    if (data.length === 0) {
      return { currentValue: 0, previousValue: 0, trend: 0, sparklineData: [] };
    }

    const current = typeof data[data.length - 1].value === 'number' ? data[data.length - 1].value : 0;
    const previous = data.length > 1 && typeof data[data.length - 2].value === 'number' ? data[data.length - 2].value : current;
    const trendValue = previous !== 0 ? ((current - previous) / previous) * 100 : 0;

    const sparkline = data.slice(-20).map((m, i) => ({
      index: i,
      value: typeof m.value === 'number' ? m.value : 0
    }));

    return {
      currentValue: current,
      previousValue: previous,
      trend: trendValue,
      sparklineData: sparkline
    };
  }, [data]);

  const formatValue = (value: number): string => {
    const decimals = config?.decimals !== undefined ? config.decimals : 2;

    switch (config?.format) {
      case 'bytes':
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = value;
        let unitIndex = 0;
        while (size >= 1024 && unitIndex < units.length - 1) {
          size /= 1024;
          unitIndex++;
        }
        return `${size.toFixed(decimals)} ${units[unitIndex]}`;

      case 'duration':
        const hours = Math.floor(value / 3600);
        const minutes = Math.floor((value % 3600) / 60);
        const seconds = Math.floor(value % 60);
        return `${hours}h ${minutes}m ${seconds}s`;

      case 'percentage':
        return `${value.toFixed(decimals)}%`;

      case 'number':
      default:
        return value.toLocaleString(undefined, { maximumFractionDigits: decimals });
    }
  };

  const trendColor = trend > 0 ? '#4caf50' : trend < 0 ? '#f44336' : '#999';
  const trendIcon = trend > 0 ? '↑' : trend < 0 ? '↓' : '→';

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', color: '#999' }}>{title}</h3>
        {onRemove && (
          <button onClick={onRemove} style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '18px', color: '#f44336' }}>
            ×
          </button>
        )}
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <div style={{ fontSize: '36px', fontWeight: 'bold', marginBottom: '8px' }}>
          {formatValue(currentValue)}
        </div>

        {config?.showTrend !== false && (
          <div style={{ fontSize: '14px', color: trendColor, marginBottom: '8px' }}>
            {trendIcon} {Math.abs(trend).toFixed(2)}%
          </div>
        )}

        {config?.showSparkline !== false && sparklineData.length > 0 && (
          <ResponsiveContainer width="100%" height={60}>
            <LineChart data={sparklineData}>
              <Line
                type="monotone"
                dataKey="value"
                stroke="#2196f3"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default StatCard;
