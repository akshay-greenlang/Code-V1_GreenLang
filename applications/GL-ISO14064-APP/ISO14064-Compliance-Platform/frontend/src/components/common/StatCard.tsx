/**
 * GL-ISO14064-APP v1.0 - Stat Card Component
 *
 * Reusable KPI card with title, value, unit, optional trend
 * indicator, and subtitle.  Used on the executive dashboard.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';

interface Props {
  title: string;
  value: string | number;
  unit?: string;
  trend?: number | null;
  trendLabel?: string;
  subtitle?: string;
  color?: string;
}

const StatCard: React.FC<Props> = ({
  title,
  value,
  unit = '',
  trend = null,
  trendLabel = 'vs prior year',
  subtitle,
  color,
}) => {
  const theme = useTheme();

  const trendColor =
    trend === null || trend === undefined
      ? theme.palette.text.secondary
      : trend < 0
        ? theme.palette.success.main
        : trend > 0
          ? theme.palette.error.main
          : theme.palette.text.secondary;

  const TrendIcon =
    trend === null || trend === undefined
      ? null
      : trend < 0
        ? TrendingDownIcon
        : trend > 0
          ? TrendingUpIcon
          : TrendingFlatIcon;

  return (
    <Card>
      <CardContent>
        <Typography variant="caption" color="text.secondary" fontWeight={500} gutterBottom>
          {title}
        </Typography>
        <Box display="flex" alignItems="baseline" gap={0.5} mt={0.5}>
          <Typography
            variant="h5"
            fontWeight={700}
            sx={{ color: color ?? theme.palette.text.primary }}
          >
            {typeof value === 'number'
              ? value.toLocaleString(undefined, { maximumFractionDigits: 1 })
              : value}
          </Typography>
          {unit && (
            <Typography variant="body2" color="text.secondary">
              {unit}
            </Typography>
          )}
        </Box>
        {trend !== null && trend !== undefined && (
          <Box display="flex" alignItems="center" gap={0.5} mt={0.5}>
            {TrendIcon && <TrendIcon sx={{ fontSize: 16, color: trendColor }} />}
            <Typography variant="caption" sx={{ color: trendColor, fontWeight: 600 }}>
              {trend > 0 ? '+' : ''}
              {trend.toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {trendLabel}
            </Typography>
          </Box>
        )}
        {subtitle && (
          <Typography variant="caption" color="text.secondary" mt={0.5} display="block">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
