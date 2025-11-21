/**
 * KPI Card Component
 *
 * Displays a key performance indicator with trend and status
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { Trend } from '../../types';

export interface KPICardProps {
  title: string;
  value: number | string;
  unit?: string;
  target?: number;
  trend?: Trend;
  trendValue?: number;
  icon?: React.ReactNode;
  status?: 'good' | 'warning' | 'critical';
  loading?: boolean;
  format?: (value: number) => string;
  onClick?: () => void;
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit = '',
  target,
  trend,
  trendValue,
  icon,
  status,
  loading = false,
  format,
  onClick,
}) => {
  // Format value
  const formattedValue = typeof value === 'number' && format
    ? format(value)
    : value;

  // Get trend icon
  const getTrendIcon = () => {
    if (!trend) return null;

    const iconProps = { fontSize: 'small' as const };

    switch (trend) {
      case 'increasing':
        return <TrendingUp {...iconProps} color="success" />;
      case 'decreasing':
        return <TrendingDown {...iconProps} color="error" />;
      case 'stable':
        return <TrendingFlat {...iconProps} color="info" />;
      case 'fluctuating':
        return <TrendingUp {...iconProps} color="warning" />;
      default:
        return null;
    }
  };

  // Get status indicator
  const getStatusIcon = () => {
    if (!status) return null;

    switch (status) {
      case 'good':
        return <CheckCircle fontSize="small" color="success" />;
      case 'warning':
        return <Warning fontSize="small" color="warning" />;
      case 'critical':
        return <ErrorIcon fontSize="small" color="error" />;
      default:
        return null;
    }
  };

  // Get status color
  const getStatusColor = () => {
    switch (status) {
      case 'good':
        return 'success.main';
      case 'warning':
        return 'warning.main';
      case 'critical':
        return 'error.main';
      default:
        return 'text.primary';
    }
  };

  // Calculate achievement percentage
  const achievement = target && typeof value === 'number'
    ? Math.round((value / target) * 100)
    : undefined;

  return (
    <Card
      sx={{
        height: '100%',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': onClick ? {
          transform: 'translateY(-4px)',
          boxShadow: 4,
        } : {},
      }}
      onClick={onClick}
    >
      <CardContent>
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            mb: 2,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            {title}
          </Typography>
          {icon && (
            <Box sx={{ color: 'primary.main' }}>{icon}</Box>
          )}
        </Box>

        {/* Value */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={40} />
          </Box>
        ) : (
          <>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 0.5,
                mb: 1,
              }}
            >
              <Typography
                variant="h3"
                component="div"
                sx={{
                  fontWeight: 600,
                  color: getStatusColor(),
                }}
              >
                {formattedValue}
              </Typography>
              {unit && (
                <Typography variant="body1" color="text.secondary">
                  {unit}
                </Typography>
              )}
            </Box>

            {/* Status and Trend */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                flexWrap: 'wrap',
              }}
            >
              {getStatusIcon()}

              {trend && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5,
                  }}
                >
                  {getTrendIcon()}
                  {trendValue !== undefined && (
                    <Typography variant="body2" color="text.secondary">
                      {trendValue > 0 ? '+' : ''}
                      {trendValue.toFixed(1)}%
                    </Typography>
                  )}
                </Box>
              )}

              {achievement !== undefined && (
                <Chip
                  label={`${achievement}% of target`}
                  size="small"
                  color={
                    achievement >= 100
                      ? 'success'
                      : achievement >= 80
                      ? 'warning'
                      : 'error'
                  }
                  variant="outlined"
                />
              )}
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default KPICard;
