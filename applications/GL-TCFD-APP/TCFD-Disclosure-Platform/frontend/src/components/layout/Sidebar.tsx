/**
 * Sidebar - Navigation for the TCFD Disclosure Platform
 *
 * Organized by the four TCFD pillars plus cross-cutting tools.
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
  Collapse,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  AccountBalance,
  TrendingUp,
  Warning,
  Whatshot,
  SwapHoriz,
  EmojiObjects,
  AttachMoney,
  Shield,
  Assessment,
  Description,
  FindInPage,
  CompareArrows,
  Settings as SettingsIcon,
  ExpandLess,
  ExpandMore,
  Landscape,
  Thermostat,
} from '@mui/icons-material';

const DRAWER_WIDTH = 260;

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

interface NavGroup {
  label: string;
  icon: React.ReactNode;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Strategy',
    icon: <TrendingUp />,
    items: [
      { label: 'Climate Risks', path: '/strategy/risks', icon: <Warning /> },
      { label: 'Opportunities', path: '/strategy/opportunities', icon: <EmojiObjects /> },
      { label: 'Scenario Analysis', path: '/scenarios', icon: <Thermostat /> },
      { label: 'Physical Risk', path: '/physical-risk', icon: <Landscape /> },
      { label: 'Transition Risk', path: '/transition-risk', icon: <SwapHoriz /> },
    ],
  },
];

const TOP_NAV: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { label: 'Governance', path: '/governance', icon: <AccountBalance /> },
];

const BOTTOM_NAV: NavItem[] = [
  { label: 'Opportunities', path: '/opportunities', icon: <TrendingUp /> },
  { label: 'Financial Impact', path: '/financial-impact', icon: <AttachMoney /> },
  { label: 'Risk Management', path: '/risk-management', icon: <Shield /> },
  { label: 'Metrics & Targets', path: '/metrics-targets', icon: <Assessment /> },
];

const TOOLS_NAV: NavItem[] = [
  { label: 'Disclosure Builder', path: '/disclosure', icon: <Description /> },
  { label: 'Gap Analysis', path: '/gap-analysis', icon: <FindInPage /> },
  { label: 'ISSB Crosswalk', path: '/issb-crosswalk', icon: <CompareArrows /> },
  { label: 'Settings', path: '/settings', icon: <SettingsIcon /> },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open, onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [strategyOpen, setStrategyOpen] = React.useState(true);

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
        <Whatshot sx={{ color: 'primary.main', fontSize: 28 }} />
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'primary.main', lineHeight: 1.2 }}>
            GL-TCFD
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Climate Disclosure Platform
          </Typography>
        </Box>
      </Box>

      <Divider />

      <List dense sx={{ py: 1 }}>
        {TOP_NAV.map(renderNavItem)}
      </List>

      <Divider />

      {/* Strategy group with collapse */}
      <List dense sx={{ py: 0 }}>
        <ListItem disablePadding>
          <ListItemButton onClick={() => setStrategyOpen(!strategyOpen)} sx={{ mx: 1, borderRadius: 1 }}>
            <ListItemIcon sx={{ minWidth: 36 }}><TrendingUp /></ListItemIcon>
            <ListItemText primary="Strategy" primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 600 }} />
            {strategyOpen ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
        </ListItem>
        <Collapse in={strategyOpen} timeout="auto" unmountOnExit>
          <List dense sx={{ pl: 2 }}>
            {NAV_GROUPS[0].items.map(renderNavItem)}
          </List>
        </Collapse>
      </List>

      <Divider />

      <List dense sx={{ py: 1 }}>
        {BOTTOM_NAV.map(renderNavItem)}
      </List>

      <Divider />

      <Typography variant="overline" sx={{ px: 2, pt: 1, color: 'text.secondary', fontSize: '0.65rem' }}>
        Tools
      </Typography>
      <List dense sx={{ py: 0 }}>
        {TOOLS_NAV.map(renderNavItem)}
      </List>
    </Drawer>
  );
};

export default Sidebar;
