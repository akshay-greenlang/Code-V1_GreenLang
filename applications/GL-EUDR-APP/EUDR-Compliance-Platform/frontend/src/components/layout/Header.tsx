/**
 * Header Component
 *
 * MUI AppBar with page title, breadcrumbs, notification bell,
 * and user avatar menu.
 */

import React from 'react';
import { useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Avatar,
  Box,
  Breadcrumbs,
  Link,
} from '@mui/material';
import {
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useAppSelector } from '../../store/hooks';

const routeTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/suppliers': 'Suppliers',
  '/plots': 'Plot Registry',
  '/risk': 'Risk Assessment',
  '/dds': 'DDS Management',
  '/documents': 'Documents',
  '/pipeline': 'Pipeline',
};

interface HeaderProps {
  drawerWidth: number;
}

const Header: React.FC<HeaderProps> = ({ drawerWidth }) => {
  const location = useLocation();
  const alerts = useAppSelector((state) => state.dashboard.alerts);
  const unreadCount = alerts.filter((a) => !a.is_read).length;

  const basePath = '/' + (location.pathname.split('/')[1] || '');
  const pageTitle = routeTitles[basePath] || 'GL-EUDR';

  const pathSegments = location.pathname.split('/').filter(Boolean);

  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        width: `calc(100% - ${drawerWidth}px)`,
        ml: `${drawerWidth}px`,
        backgroundColor: '#ffffff',
        color: '#333333',
        borderBottom: '1px solid #e0e0e0',
        transition: 'width 0.2s ease-in-out, margin-left 0.2s ease-in-out',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="h6" component="h1" sx={{ fontWeight: 600 }}>
            {pageTitle}
          </Typography>
          {pathSegments.length > 1 && (
            <Breadcrumbs
              aria-label="breadcrumb"
              sx={{ fontSize: '0.75rem', mt: -0.5 }}
            >
              <Link underline="hover" color="inherit" href="/">
                Home
              </Link>
              {pathSegments.map((segment, idx) => (
                <Typography
                  key={idx}
                  variant="caption"
                  color={
                    idx === pathSegments.length - 1
                      ? 'text.primary'
                      : 'inherit'
                  }
                  sx={{ textTransform: 'capitalize' }}
                >
                  {segment.replace(/-/g, ' ')}
                </Typography>
              ))}
            </Breadcrumbs>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton color="inherit" aria-label="notifications">
            <Badge badgeContent={unreadCount} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>

          <IconButton color="inherit" aria-label="settings" href="/settings">
            <SettingsIcon />
          </IconButton>

          <Avatar
            sx={{
              width: 36,
              height: 36,
              ml: 1,
              bgcolor: '#2e7d32',
              fontSize: '0.9rem',
            }}
          >
            EU
          </Avatar>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
