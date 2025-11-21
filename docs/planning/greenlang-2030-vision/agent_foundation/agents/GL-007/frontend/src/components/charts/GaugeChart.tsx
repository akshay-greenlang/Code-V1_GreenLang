/**
 * Gauge Chart Component
 *
 * Circular gauge for displaying real-time KPI values
 */

import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import { Chart as ChartJS, ArcElement, Tooltip } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip);

export interface GaugeChartProps {
  value: number;
  maxValue: number;
  minValue?: number;
  title?: string;
  unit?: string;
  thresholds?: {
    good?: number;
    warning?: number;
    critical?: number;
  };
  size?: number;
  showValue?: boolean;
  animated?: boolean;
}

const GaugeChart: React.FC<GaugeChartProps> = ({
  value,
  maxValue,
  minValue = 0,
  title,
  unit = '',
  thresholds = { good: 80, warning: 60, critical: 40 },
  size = 200,
  showValue = true,
  animated = true,
}) => {
  const theme = useTheme();

  // Normalize value to percentage
  const range = maxValue - minValue;
  const normalizedValue = ((value - minValue) / range) * 100;

  // Determine color based on thresholds
  const getColor = () => {
    if (thresholds.good && normalizedValue >= thresholds.good) {
      return theme.palette.success.main;
    }
    if (thresholds.warning && normalizedValue >= thresholds.warning) {
      return theme.palette.warning.main;
    }
    return theme.palette.error.main;
  };

  const color = getColor();

  // Chart data
  const data = {
    datasets: [
      {
        data: [normalizedValue, 100 - normalizedValue],
        backgroundColor: [color, theme.palette.grey[200]],
        borderWidth: 0,
        cutout: '80%',
        circumference: 180,
        rotation: 270,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      tooltip: { enabled: false },
      legend: { display: false },
    },
    animation: animated ? {
      animateRotate: true,
      animateScale: true,
    } : false,
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 1,
      }}
    >
      {title && (
        <Typography variant="body2" color="text.secondary" align="center">
          {title}
        </Typography>
      )}

      <Box
        sx={{
          position: 'relative',
          width: size,
          height: size / 2 + 20,
        }}
      >
        <Doughnut data={data} options={options} />

        {showValue && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: '50%',
              transform: 'translateX(-50%)',
              textAlign: 'center',
            }}
          >
            <Typography
              variant="h4"
              sx={{
                fontWeight: 600,
                color: color,
              }}
            >
              {value.toFixed(1)}
            </Typography>
            {unit && (
              <Typography variant="caption" color="text.secondary">
                {unit}
              </Typography>
            )}
          </Box>
        )}
      </Box>

      {/* Threshold indicators */}
      <Box
        sx={{
          display: 'flex',
          gap: 2,
          mt: 1,
        }}
      >
        {thresholds.good !== undefined && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: theme.palette.success.main,
              }}
            />
            <Typography variant="caption">≥{thresholds.good}%</Typography>
          </Box>
        )}
        {thresholds.warning !== undefined && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: theme.palette.warning.main,
              }}
            />
            <Typography variant="caption">≥{thresholds.warning}%</Typography>
          </Box>
        )}
        {thresholds.critical !== undefined && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: theme.palette.error.main,
              }}
            />
            <Typography variant="caption">&lt;{thresholds.warning}%</Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default GaugeChart;
