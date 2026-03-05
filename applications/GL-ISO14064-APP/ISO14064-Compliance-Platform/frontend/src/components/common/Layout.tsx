/**
 * Layout - Main application shell
 *
 * Combines a permanent sidebar navigation drawer, a top AppBar,
 * and a content area.  Follows the GreenLang design language
 * with the green (#1b5e20) brand accent.
 *
 * Sidebar navigation covers all ISO 14064-1 platform sections:
 *   Dashboard, Organizations, Inventories, Verification,
 *   Reports, Management, Quality, Settings
 */

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Tooltip,
  IconButton,
  Avatar,
} from '@mui/material';
import {
  Dashboard,
  Business,
  Inventory2,
  VerifiedUser,
  Assessment,
  ManageAccounts,
  HighQuality,
  Settings,
  CompareArrows,
  Notifications,
} from '@mui/icons-material';

export const DRAWER_WIDTH = 260;

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactElement;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: <Dashboard />,
    description: 'Executive overview of ISO 14064-1 inventory',
  },
  {
    path: '/organizations',
    label: 'Organizations',
    icon: <Business />,
    description: 'Organization setup and boundary configuration',
  },
  {
    path: '/inventories',
    label: 'Inventories',
    icon: <Inventory2 />,
    description: 'GHG inventory management by reporting year',
  },
  {
    path: '/verification',
    label: 'Verification',
    icon: <VerifiedUser />,
    description: 'ISO 14064-3 verification workflow',
  },
  {
    path: '/reports',
    label: 'Reports',
    icon: <Assessment />,
    description: 'ISO 14064-1 report generation and compliance',
  },
  {
    path: '/management',
    label: 'Management Plan',
    icon: <ManageAccounts />,
    description: 'GHG management actions and improvement plan',
  },
  {
    path: '/quality',
    label: 'Data Quality',
    icon: <HighQuality />,
    description: 'Data quality scorecard and procedures',
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: <Settings />,
    description: 'Platform configuration and preferences',
  },
];

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string): boolean => {
    if (path === '/dashboard') {
      return location.pathname === '/dashboard' || location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f5f7f5' }}>
      {/* Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            backgroundColor: '#fafbfa',
            borderRight: 'none',
            boxShadow: '2px 0 8px rgba(0, 0, 0, 0.04)',
          },
        }}
      >
        {/* Branding */}
        <Box sx={{ p: 2.5, display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              width: 38,
              height: 38,
              borderRadius: 2,
              background: 'linear-gradient(135deg, #1b5e20, #2e7d32)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontWeight: 700,
              fontSize: 14,
              letterSpacing: 0.5,
              boxShadow: '0 2px 6px rgba(27, 94, 32, 0.3)',
            }}
          >
            GL
          </Box>
          <Box>
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: 700, lineHeight: 1.2, color: '#1a1a2e' }}
            >
              GreenLang
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ lineHeight: 1.2 }}
            >
              ISO 14064-1 Platform
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ mx: 2, opacity: 0.6 }} />

        {/* Navigation */}
        <List sx={{ px: 1.5, pt: 1.5, flexGrow: 1 }}>
          {NAV_ITEMS.map((item) => {
            const active = isActive(item.path);
            return (
              <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
                <Tooltip title={item.description} placement="right" arrow>
                  <ListItemButton
                    onClick={() => navigate(item.path)}
                    selected={active}
                    sx={{
                      borderRadius: 2,
                      py: 1,
                      '&.Mui-selected': {
                        backgroundColor: 'rgba(27, 94, 32, 0.08)',
                        '& .MuiListItemIcon-root': { color: '#1b5e20' },
                        '& .MuiListItemText-primary': {
                          fontWeight: 600,
                          color: '#1b5e20',
                        },
                      },
                      '&:hover': {
                        backgroundColor: 'rgba(27, 94, 32, 0.04)',
                      },
                      transition: 'background-color 0.15s ease',
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        minWidth: 40,
                        color: active ? '#1b5e20' : 'text.secondary',
                      }}
                    >
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText
                      primary={item.label}
                      primaryTypographyProps={{
                        variant: 'body2',
                        fontWeight: active ? 600 : 400,
                      }}
                    />
                  </ListItemButton>
                </Tooltip>
              </ListItem>
            );
          })}
        </List>

        {/* Footer */}
        <Divider sx={{ mx: 2, opacity: 0.6 }} />
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ fontSize: '0.7rem' }}
          >
            ISO 14064-1:2018 Compliance
          </Typography>
          <Typography
            variant="caption"
            display="block"
            color="text.secondary"
            sx={{ fontSize: '0.65rem', mt: 0.25 }}
          >
            v1.0
          </Typography>
        </Box>
      </Drawer>

      {/* Top AppBar */}
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
            ISO 14064-1 Compliance Platform
          </Typography>
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

      {/* Content area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          ml: `${DRAWER_WIDTH}px`,
          p: 3,
          minHeight: '100vh',
          maxWidth: `calc(100% - ${DRAWER_WIDTH}px)`,
          overflow: 'auto',
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
