/**
 * StatCard - Reusable KPI metric display card
 *
 * Displays a labeled metric value with optional change indicator
 * and icon. Used across dashboard and scope pages for key statistics.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material';

interface StatCardProps {
  title: string;
  value: string;
  subtitle?: string;
  change?: number;
  changeLabel?: string;
  icon?: React.ReactNode;
  color?: string;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  change,
  changeLabel,
  icon,
  color = '#1b5e20',
}) => {
  const getChangeColor = (): 'success' | 'error' | 'default' => {
    if (change === undefined || change === 0) return 'default';
    return change < 0 ? 'success' : 'error';
  };

  const getChangeIcon = () => {
    if (change === undefined || change === 0) return <Remove fontSize="small" />;
    return change < 0 ? <TrendingDown fontSize="small" /> : <TrendingUp fontSize="small" />;
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 700, color }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {subtitle}
              </Typography>
            )}
          </Box>
          {icon && (
            <Box
              sx={{
                p: 1,
                borderRadius: 2,
                backgroundColor: `${color}14`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {icon}
            </Box>
          )}
        </Box>
        {change !== undefined && (
          <Box sx={{ mt: 1.5 }}>
            <Chip
              icon={getChangeIcon()}
              label={changeLabel || `${change > 0 ? '+' : ''}${change.toFixed(1)}% YoY`}
              size="small"
              color={getChangeColor()}
              variant="outlined"
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
