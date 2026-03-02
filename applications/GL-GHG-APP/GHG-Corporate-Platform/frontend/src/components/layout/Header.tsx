/**
 * Header - Top application bar
 *
 * Displays:
 *   - Organization name (from Redux store)
 *   - Reporting year selector (last 10 years)
 *   - Notification bell with unread badge
 *   - Settings icon
 *   - User avatar
 */

import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  IconButton,
  Badge,
  Avatar,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Menu,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Notifications,
  Settings,
  Logout,
  Person,
} from '@mui/icons-material';
import { DRAWER_WIDTH } from './Sidebar';
import { useAppSelector } from '../../store/hooks';

const ROUTE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/setup': 'Inventory Setup',
  '/scope1': 'Scope 1 - Direct Emissions',
  '/scope2': 'Scope 2 - Indirect Emissions',
  '/scope3': 'Scope 3 - Value Chain',
  '/reports': 'Reports & Disclosure',
  '/targets': 'Reduction Targets',
  '/verification': 'Verification',
};

interface HeaderProps {
  reportingYear?: number;
  onYearChange?: (year: number) => void;
}

const Header: React.FC<HeaderProps> = ({
  reportingYear: propYear,
  onYearChange: propOnChange,
}) => {
  const location = useLocation();
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 10 }, (_, i) => currentYear - i);

  const [localYear, setLocalYear] = useState(currentYear - 1);
  const reportingYear = propYear ?? localYear;
  const onYearChange = propOnChange ?? setLocalYear;

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const orgName = useAppSelector(
    (state) => state.inventory.organization?.name ?? 'GreenLang Corp',
  );
  const alertCount = useAppSelector(
    (state) => state.dashboard.alerts.filter((a) => !a.is_read).length,
  );

  const title = ROUTE_TITLES[location.pathname] || 'GHG Corporate Platform';

  const handleYearChange = (event: SelectChangeEvent<number>) => {
    onYearChange(event.target.value as number);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  return (
    <AppBar
      position="fixed"
      color="inherit"
      elevation={0}
      sx={{
        width: `calc(100% - ${DRAWER_WIDTH}px)`,
        ml: `${DRAWER_WIDTH}px`,
        borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
        backgroundColor: '#ffffff',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between', minHeight: 64 }}>
        {/* Left: Page title + org name */}
        <Box>
          <Typography variant="h5" color="text.primary" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {orgName}
          </Typography>
        </Box>

        {/* Right: Controls */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          {/* Reporting year selector */}
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="reporting-year-label">Reporting Year</InputLabel>
            <Select
              labelId="reporting-year-label"
              value={reportingYear}
              label="Reporting Year"
              onChange={handleYearChange}
              sx={{ '& .MuiSelect-select': { py: 0.75 } }}
            >
              {years.map((y) => (
                <MenuItem key={y} value={y}>
                  {y}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Notification bell */}
          <IconButton color="default" aria-label="Notifications">
            <Badge badgeContent={alertCount} color="error" max={99}>
              <Notifications />
            </Badge>
          </IconButton>

          {/* Settings */}
          <IconButton color="default" aria-label="Settings">
            <Settings />
          </IconButton>

          {/* User avatar */}
          <IconButton onClick={handleMenuOpen} sx={{ p: 0 }}>
            <Avatar
              sx={{
                width: 34,
                height: 34,
                bgcolor: '#1b5e20',
                fontSize: 14,
                fontWeight: 600,
              }}
            >
              GL
            </Avatar>
          </IconButton>

          {/* User menu */}
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={handleMenuClose}>
              <ListItemIcon><Person fontSize="small" /></ListItemIcon>
              <ListItemText>Profile</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleMenuClose}>
              <ListItemIcon><Logout fontSize="small" /></ListItemIcon>
              <ListItemText>Sign Out</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
