/**
 * Sidebar - Main navigation drawer
 *
 * Persistent left sidebar with 8 navigation items covering all
 * GHG Protocol Corporate Standard platform sections.
 * Uses GreenLang green theme (#1b5e20) with active route highlighting.
 *
 * Nav items:
 *   Dashboard, Inventory Setup, Scope 1, Scope 2, Scope 3,
 *   Reports, Targets, Verification
 */

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Dashboard,
  Settings as SettingsIcon,
  Factory,
  ElectricBolt,
  AccountTree,
  Assessment,
  TrackChanges,
  VerifiedUser,
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
    path: '/',
    label: 'Dashboard',
    icon: <Dashboard />,
    description: 'Executive overview of GHG inventory',
  },
  {
    path: '/setup',
    label: 'Inventory Setup',
    icon: <SettingsIcon />,
    description: 'Organization, entities, and boundaries',
  },
  {
    path: '/scope1',
    label: 'Scope 1',
    icon: <Factory />,
    description: 'Direct GHG emissions',
  },
  {
    path: '/scope2',
    label: 'Scope 2',
    icon: <ElectricBolt />,
    description: 'Indirect energy emissions',
  },
  {
    path: '/scope3',
    label: 'Scope 3',
    icon: <AccountTree />,
    description: 'Value chain emissions',
  },
  {
    path: '/reports',
    label: 'Reports',
    icon: <Assessment />,
    description: 'Report generation and disclosure',
  },
  {
    path: '/targets',
    label: 'Targets',
    icon: <TrackChanges />,
    description: 'Reduction targets and SBTi alignment',
  },
  {
    path: '/verification',
    label: 'Verification',
    icon: <VerifiedUser />,
    description: 'Third-party verification workflow',
  },
];

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string): boolean => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
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
          <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.2, color: '#1a1a2e' }}>
            GreenLang
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.2 }}>
            GHG Protocol Platform
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
                  <ListItemIcon sx={{ minWidth: 40, color: active ? '#1b5e20' : 'text.secondary' }}>
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
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
          GHG Protocol Corporate Standard
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
          v1.0
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
