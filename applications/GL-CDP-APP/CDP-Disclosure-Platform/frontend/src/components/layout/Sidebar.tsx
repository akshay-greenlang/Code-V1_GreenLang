/**
 * Sidebar - CDP navigation drawer
 *
 * Permanent sidebar with navigation for all CDP platform sections:
 * Dashboard, Questionnaire, Scoring, Gap Analysis, Benchmarking,
 * Supply Chain, Transition Plan, Verification, Reports, Historical, Settings.
 */

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Dashboard,
  QuestionAnswer,
  Speed,
  FindInPage,
  Leaderboard,
  LocalShipping,
  TrendingUp,
  VerifiedUser,
  Description,
  History,
  Settings,
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
    description: 'Executive overview with predicted CDP score',
  },
  {
    path: '/questionnaire',
    label: 'Questionnaire',
    icon: <QuestionAnswer />,
    description: 'CDP Climate Change questionnaire (13 modules)',
  },
  {
    path: '/scoring',
    label: 'Scoring Simulator',
    icon: <Speed />,
    description: 'Predict CDP score and run what-if scenarios',
  },
  {
    path: '/gaps',
    label: 'Gap Analysis',
    icon: <FindInPage />,
    description: 'Identify gaps and improvement recommendations',
  },
  {
    path: '/benchmarking',
    label: 'Benchmarking',
    icon: <Leaderboard />,
    description: 'Compare against sector peers and A-list rate',
  },
  {
    path: '/supply-chain',
    label: 'Supply Chain',
    icon: <LocalShipping />,
    description: 'Supplier engagement and emissions tracking',
  },
  {
    path: '/transition',
    label: 'Transition Plan',
    icon: <TrendingUp />,
    description: '1.5C pathway, milestones, SBTi alignment',
  },
  {
    path: '/verification',
    label: 'Verification',
    icon: <VerifiedUser />,
    description: 'Third-party verification status tracking',
  },
  {
    path: '/reports',
    label: 'Reports',
    icon: <Description />,
    description: 'Generate PDF, Excel, XML for CDP ORS',
  },
  {
    path: '/historical',
    label: 'Historical',
    icon: <History />,
    description: 'Year-over-year score comparison',
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: <Settings />,
    description: 'Organization profile and team management',
  },
];

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string): boolean => {
    if (path === '/dashboard') {
      return location.pathname === '/dashboard' || location.pathname === '/';
    }
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
            CDP Disclosure Platform
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
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
          CDP Climate Change 2025/2026
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
          v1.0
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
