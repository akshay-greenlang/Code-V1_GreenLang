/**
 * Header - Top application bar
 *
 * Displays the platform title, predicted CDP score indicator,
 * submission deadline countdown, notification bell, and user avatar.
 */

import React from 'react';
import { AppBar, Toolbar, Typography, Box, Chip, IconButton, Avatar } from '@mui/material';
import { Notifications, AccessTime } from '@mui/icons-material';
import { DRAWER_WIDTH } from './Sidebar';
import { useAppSelector } from '../../store/hooks';
import { getScoringLevelColor } from '../../utils/formatters';

const Header: React.FC = () => {
  const dashboardData = useAppSelector((s) => s.dashboard.data);

  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        ml: `${DRAWER_WIDTH}px`,
        width: `calc(100% - ${DRAWER_WIDTH}px)`,
        backgroundColor: '#fff',
        borderBottom: '1px solid #e0e0e0',
      }}
    >
      <Toolbar>
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, color: '#1a1a2e', fontWeight: 600 }}
        >
          CDP Climate Change Disclosure
        </Typography>

        {/* Score indicator */}
        {dashboardData && (
          <Chip
            label={`Score: ${dashboardData.predicted_level} (${dashboardData.predicted_score.toFixed(0)}%)`}
            sx={{
              mr: 2,
              fontWeight: 700,
              backgroundColor: getScoringLevelColor(dashboardData.predicted_level) + '15',
              color: getScoringLevelColor(dashboardData.predicted_level),
              border: `1px solid ${getScoringLevelColor(dashboardData.predicted_level)}40`,
            }}
          />
        )}

        {/* Deadline indicator */}
        {dashboardData && (
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
            <AccessTime fontSize="small" sx={{ color: 'text.secondary', mr: 0.5 }} />
            <Typography variant="caption" color="text.secondary">
              {dashboardData.days_until_deadline > 0
                ? `${dashboardData.days_until_deadline}d to deadline`
                : 'Deadline passed'}
            </Typography>
          </Box>
        )}

        <IconButton size="small" sx={{ mr: 1 }}>
          <Notifications fontSize="small" />
        </IconButton>
        <Avatar
          sx={{
            width: 32,
            height: 32,
            bgcolor: '#1b5e20',
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          U
        </Avatar>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
