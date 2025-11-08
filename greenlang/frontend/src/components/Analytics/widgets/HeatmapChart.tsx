/**
 * Heatmap chart widget for correlation matrices and multi-dimensional data.
 */

import React, { useMemo } from 'react';
import { Metric } from '../MetricService';

interface HeatmapChartProps {
  title: string;
  data: Metric[];
  config?: {
    colorScale?: { min: string; mid: string; max: string };
    showValues?: boolean;
  };
  onRemove?: () => void;
}

const HeatmapChart: React.FC<HeatmapChartProps> = ({ title, data, config, onRemove }) => {
  const { matrix, minValue, maxValue, labels } = useMemo(() => {
    if (data.length === 0) {
      return { matrix: [], minValue: 0, maxValue: 0, labels: [] };
    }

    // Assume data contains a matrix in the value field
    const firstMetric = data[data.length - 1];
    const matrixData = Array.isArray(firstMetric.value) ? firstMetric.value : [[0]];

    let min = Infinity;
    let max = -Infinity;

    matrixData.forEach(row => {
      if (Array.isArray(row)) {
        row.forEach(val => {
          if (typeof val === 'number') {
            min = Math.min(min, val);
            max = Math.max(max, val);
          }
        });
      }
    });

    const rowLabels = matrixData.map((_, i) => `Row ${i + 1}`);

    return { matrix: matrixData, minValue: min, maxValue: max, labels: rowLabels };
  }, [data]);

  const getColor = (value: number) => {
    const colors = config?.colorScale || { min: '#2196f3', mid: '#ffffff', max: '#f44336' };
    const range = maxValue - minValue;

    if (range === 0) return colors.mid;

    const normalized = (value - minValue) / range;

    if (normalized < 0.5) {
      const t = normalized * 2;
      return interpolateColor(colors.min, colors.mid, t);
    } else {
      const t = (normalized - 0.5) * 2;
      return interpolateColor(colors.mid, colors.max, t);
    }
  };

  const interpolateColor = (color1: string, color2: string, t: number) => {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);

    const r = Math.round(c1.r + (c2.r - c1.r) * t);
    const g = Math.round(c1.g + (c2.g - c1.g) * t);
    const b = Math.round(c1.b + (c2.b - c1.b) * t);

    return `rgb(${r}, ${g}, ${b})`;
  };

  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  };

  const cellSize = 40;

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

      <div style={{ flex: 1, overflow: 'auto', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <svg width={matrix[0]?.length * cellSize || cellSize} height={matrix.length * cellSize}>
          {matrix.map((row, i) => (
            Array.isArray(row) ? row.map((cell, j) => (
              <g key={`${i}-${j}`}>
                <rect
                  x={j * cellSize}
                  y={i * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={typeof cell === 'number' ? getColor(cell) : '#ccc'}
                  stroke="#fff"
                  strokeWidth={1}
                />
                {config?.showValues !== false && typeof cell === 'number' && (
                  <text
                    x={j * cellSize + cellSize / 2}
                    y={i * cellSize + cellSize / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="#000"
                    fontSize="10px"
                  >
                    {cell.toFixed(1)}
                  </text>
                )}
              </g>
            )) : null
          ))}
        </svg>
      </div>

      {/* Color scale legend */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '8px' }}>
        <div style={{ marginRight: '8px' }}>{minValue.toFixed(2)}</div>
        <div style={{ width: '200px', height: '20px', background: `linear-gradient(to right, ${config?.colorScale?.min || '#2196f3'}, ${config?.colorScale?.mid || '#ffffff'}, ${config?.colorScale?.max || '#f44336'})` }} />
        <div style={{ marginLeft: '8px' }}>{maxValue.toFixed(2)}</div>
      </div>

      {data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>No data available</div>
      )}
    </div>
  );
};

export default HeatmapChart;
