/**
 * Header - Top app bar for the EU Taxonomy Alignment Platform.
 */

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Breadcrumbs,
  Link,
  Box,
  Chip,
} from '@mui/material';
import { Menu as MenuIcon, AccountCircle, Notifications } from '@mui/icons-material';
import { useLocation, Link as RouterLink } from 'react-router-dom';

interface HeaderProps {
  onMenuToggle: () => void;
}

const routeLabels: Record<string, string> = {
  '/': 'Dashboard',
  '/screening': 'Activity Screening',
  '/substantial-contribution': 'Substantial Contribution',
  '/dnsh': 'DNSH Assessment',
  '/safeguards': 'Minimum Safeguards',
  '/kpi': 'KPI Calculator',
  '/gar': 'GAR Calculator',
  '/alignment': 'Alignment Workflow',
  '/reporting': 'Reporting',
  '/portfolio': 'Portfolio Management',
  '/data-quality': 'Data Quality',
  '/gap-analysis': 'Gap Analysis',
  '/regulatory': 'Regulatory Updates',
  '/settings': 'Settings',
};

const Header: React.FC<HeaderProps> = ({ onMenuToggle }) => {
  const location = useLocation();
  const currentLabel = routeLabels[location.pathname] || 'Dashboard';

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: '#FFFFFF',
        color: '#212121',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      }}
    >
      <Toolbar>
        <IconButton edge="start" onClick={onMenuToggle} sx={{ mr: 2 }}>
          <MenuIcon />
        </IconButton>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexGrow: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main' }}>
            GL-Taxonomy-APP
          </Typography>
          <Chip label="v1.0" size="small" color="primary" variant="outlined" />

          <Box sx={{ mx: 2, color: '#BDBDBD' }}>|</Box>

          <Breadcrumbs aria-label="breadcrumb" sx={{ fontSize: '0.875rem' }}>
            <Link component={RouterLink} to="/" underline="hover" color="inherit">
              Home
            </Link>
            <Typography color="text.primary" sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
              {currentLabel}
            </Typography>
          </Breadcrumbs>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton>
            <Notifications />
          </IconButton>
          <IconButton>
            <AccountCircle />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
