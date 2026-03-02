/**
 * StatCard Component
 *
 * Reusable KPI card displaying an icon, title, primary value,
 * and optional subtitle with color variant support.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, SxProps, Theme } from '@mui/material';

interface StatCardProps {
  icon: React.ReactElement;
  title: string;
  value: string | number;
  subtitle?: string;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  sx?: SxProps<Theme>;
}

const colorMap: Record<string, string> = {
  primary: '#1976d2',
  secondary: '#9c27b0',
  success: '#2e7d32',
  warning: '#ed6c02',
  error: '#d32f2f',
  info: '#0288d1',
};

const bgMap: Record<string, string> = {
  primary: 'rgba(25, 118, 210, 0.08)',
  secondary: 'rgba(156, 39, 176, 0.08)',
  success: 'rgba(46, 125, 50, 0.08)',
  warning: 'rgba(237, 108, 2, 0.08)',
  error: 'rgba(211, 47, 47, 0.08)',
  info: 'rgba(2, 136, 209, 0.08)',
};

const StatCard: React.FC<StatCardProps> = ({
  icon,
  title,
  value,
  subtitle,
  color = 'primary',
  sx,
}) => {
  const iconColor = colorMap[color] || colorMap.primary;
  const iconBg = bgMap[color] || bgMap.primary;

  return (
    <Card sx={{ height: '100%', ...sx }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
          <Box
            sx={{
              p: 1,
              borderRadius: 2,
              backgroundColor: iconBg,
              color: iconColor,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
          <Box sx={{ flexGrow: 1 }}>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 0.5 }}
            >
              {title}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5, display: 'block' }}
              >
                {subtitle}
              </Typography>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default StatCard;
