/**
 * StatCard - KPI card component with icon, value, trend indicator, and color theming.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: number;
  trendLabel?: string;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'success' | 'info';
  format?: 'number' | 'currency' | 'percent';
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  trendLabel,
  color = 'primary',
  format,
}) => {
  const formattedValue = React.useMemo(() => {
    if (typeof value === 'string') return value;
    switch (format) {
      case 'currency':
        return value >= 1_000_000
          ? `$${(value / 1_000_000).toFixed(1)}M`
          : value >= 1_000
          ? `$${(value / 1_000).toFixed(0)}K`
          : `$${value.toLocaleString()}`;
      case 'percent':
        return `${value.toFixed(1)}%`;
      default:
        return value.toLocaleString();
    }
  }, [value, format]);

  const trendColor = trend !== undefined ? (trend > 0 ? 'success.main' : trend < 0 ? 'error.main' : 'text.secondary') : undefined;

  const TrendIcon = trend !== undefined ? (trend > 0 ? TrendingUp : trend < 0 ? TrendingDown : TrendingFlat) : null;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            {title}
          </Typography>
          <Box sx={{ color: `${color}.main`, opacity: 0.8 }}>
            {icon}
          </Box>
        </Box>

        <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
          {formattedValue}
        </Typography>

        {(trend !== undefined || subtitle) && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {TrendIcon && (
              <TrendIcon sx={{ fontSize: 18, color: trendColor }} />
            )}
            {trend !== undefined && (
              <Typography variant="body2" sx={{ color: trendColor, fontWeight: 500 }}>
                {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
              </Typography>
            )}
            {trendLabel && (
              <Typography variant="body2" sx={{ color: 'text.secondary', ml: 0.5 }}>
                {trendLabel}
              </Typography>
            )}
            {subtitle && !trend && (
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                {subtitle}
              </Typography>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
