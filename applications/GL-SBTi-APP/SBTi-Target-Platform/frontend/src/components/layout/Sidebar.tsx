/**
 * Sidebar - Navigation for the SBTi Target Validation Platform.
 *
 * Organized by target setting, monitoring, and tools sections.
 */

import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrackChanges,
  Timeline,
  FactCheck,
  TrendingDown,
  Thermostat,
  AccountTree,
  Nature,
  AccountBalance,
  Autorenew,
  Description,
  CompareArrows,
  FindInPage,
  Settings as SettingsIcon,
  GpsFixed,
} from '@mui/icons-material';

const DRAWER_WIDTH = 260;

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

const TARGET_NAV: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { label: 'Target Configuration', path: '/targets', icon: <TrackChanges /> },
  { label: 'Pathway Calculator', path: '/pathways', icon: <Timeline /> },
  { label: 'Validation Checker', path: '/validation', icon: <FactCheck /> },
  { label: 'Scope 3 Screening', path: '/scope3', icon: <AccountTree /> },
  { label: 'FLAG Assessment', path: '/flag', icon: <Nature /> },
];

const MONITORING_NAV: NavItem[] = [
  { label: 'Progress Tracking', path: '/progress', icon: <TrendingDown /> },
  { label: 'Temperature Scoring', path: '/temperature', icon: <Thermostat /> },
  { label: 'Recalculation & Review', path: '/recalculation', icon: <Autorenew /> },
  { label: 'Financial Institutions', path: '/fi', icon: <AccountBalance /> },
];

const TOOLS_NAV: NavItem[] = [
  { label: 'Reports', path: '/reports', icon: <Description /> },
  { label: 'Framework Alignment', path: '/frameworks', icon: <CompareArrows /> },
  { label: 'Gap Analysis', path: '/gap-analysis', icon: <FindInPage /> },
  { label: 'Settings', path: '/settings', icon: <SettingsIcon /> },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open, onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavigate = (path: string) => {
    navigate(path);
    onClose();
  };

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  const renderNavItem = (item: NavItem) => (
    <ListItem key={item.path} disablePadding>
      <ListItemButton
        selected={isActive(item.path)}
        onClick={() => handleNavigate(item.path)}
        sx={{
          borderRadius: 1,
          mx: 1,
          '&.Mui-selected': {
            backgroundColor: 'primary.main',
            color: 'white',
            '& .MuiListItemIcon-root': { color: 'white' },
            '&:hover': { backgroundColor: 'primary.dark' },
          },
        }}
      >
        <ListItemIcon sx={{ minWidth: 36 }}>{item.icon}</ListItemIcon>
        <ListItemText
          primary={item.label}
          primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: isActive(item.path) ? 600 : 400 }}
        />
      </ListItemButton>
    </ListItem>
  );

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          borderRight: '1px solid #E0E0E0',
          backgroundColor: '#FAFAFA',
        },
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <GpsFixed sx={{ color: 'primary.main', fontSize: 28 }} />
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'primary.main', lineHeight: 1.2 }}>
            GL-SBTi
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Target Validation Platform
          </Typography>
        </Box>
      </Box>

      <Divider />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Target Setting
      </Typography>
      <List dense sx={{ py: 0 }}>
        {TARGET_NAV.map(renderNavItem)}
      </List>

      <Divider sx={{ mt: 1 }} />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Monitoring
      </Typography>
      <List dense sx={{ py: 0 }}>
        {MONITORING_NAV.map(renderNavItem)}
      </List>

      <Divider sx={{ mt: 1 }} />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Tools
      </Typography>
      <List dense sx={{ py: 0 }}>
        {TOOLS_NAV.map(renderNavItem)}
      </List>
    </Drawer>
  );
};

export default Sidebar;
