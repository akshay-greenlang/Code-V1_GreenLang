/**
 * Sidebar - Navigation for the EU Taxonomy Alignment Platform.
 *
 * Organized in 4 groups: Overview, Assessment, Calculation, Analytics.
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
  Search,
  CheckCircle,
  DoNotDisturb,
  Shield,
  Calculate,
  AccountBalance,
  LinearScale,
  Description,
  FolderSpecial,
  Assessment,
  FindInPage,
  Gavel,
  Settings as SettingsIcon,
  Eco,
} from '@mui/icons-material';

const DRAWER_WIDTH = 260;

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

const OVERVIEW_NAV: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
];

const ASSESSMENT_NAV: NavItem[] = [
  { label: 'Activity Screening', path: '/screening', icon: <Search /> },
  { label: 'Substantial Contribution', path: '/substantial-contribution', icon: <CheckCircle /> },
  { label: 'DNSH Assessment', path: '/dnsh', icon: <DoNotDisturb /> },
  { label: 'Minimum Safeguards', path: '/safeguards', icon: <Shield /> },
  { label: 'Alignment Workflow', path: '/alignment', icon: <LinearScale /> },
];

const CALCULATION_NAV: NavItem[] = [
  { label: 'KPI Calculator', path: '/kpi', icon: <Calculate /> },
  { label: 'GAR Calculator', path: '/gar', icon: <AccountBalance /> },
  { label: 'Reporting', path: '/reporting', icon: <Description /> },
  { label: 'Portfolio Management', path: '/portfolio', icon: <FolderSpecial /> },
];

const ANALYTICS_NAV: NavItem[] = [
  { label: 'Data Quality', path: '/data-quality', icon: <Assessment /> },
  { label: 'Gap Analysis', path: '/gap-analysis', icon: <FindInPage /> },
  { label: 'Regulatory Updates', path: '/regulatory', icon: <Gavel /> },
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
        <Eco sx={{ color: 'primary.main', fontSize: 28 }} />
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'primary.main', lineHeight: 1.2 }}>
            GL-Taxonomy
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            EU Taxonomy Alignment Platform
          </Typography>
        </Box>
      </Box>

      <Divider />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Overview
      </Typography>
      <List dense sx={{ py: 0 }}>
        {OVERVIEW_NAV.map(renderNavItem)}
      </List>

      <Divider sx={{ mt: 1 }} />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Assessment
      </Typography>
      <List dense sx={{ py: 0 }}>
        {ASSESSMENT_NAV.map(renderNavItem)}
      </List>

      <Divider sx={{ mt: 1 }} />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Calculation
      </Typography>
      <List dense sx={{ py: 0 }}>
        {CALCULATION_NAV.map(renderNavItem)}
      </List>

      <Divider sx={{ mt: 1 }} />

      <Typography variant="overline" sx={{ px: 2, pt: 1.5, color: 'text.secondary', fontSize: '0.65rem' }}>
        Analytics
      </Typography>
      <List dense sx={{ py: 0 }}>
        {ANALYTICS_NAV.map(renderNavItem)}
      </List>
    </Drawer>
  );
};

export default Sidebar;
