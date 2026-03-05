/**
 * Header - Top application bar with organization/year selectors and notifications.
 */

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Select,
  MenuItem,
  Badge,
  Box,
  Tooltip,
  Avatar,
  SelectChangeEvent,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  HelpOutline,
  Download,
} from '@mui/icons-material';

interface HeaderProps {
  onToggleSidebar: () => void;
  orgId: string;
  onOrgChange: (orgId: string) => void;
  reportingYear: number;
  onYearChange: (year: number) => void;
}

const YEARS = [2026, 2025, 2024, 2023, 2022];

const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  orgId,
  onOrgChange,
  reportingYear,
  onYearChange,
}) => {
  const handleYearChange = (e: SelectChangeEvent<number>) => {
    onYearChange(Number(e.target.value));
  };

  const handleOrgChange = (e: SelectChangeEvent<string>) => {
    onOrgChange(e.target.value);
  };

  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        backgroundColor: 'white',
        borderBottom: '1px solid #E0E0E0',
        color: 'text.primary',
        zIndex: (theme) => theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar sx={{ gap: 2 }}>
        <IconButton edge="start" onClick={onToggleSidebar} sx={{ color: 'text.primary' }}>
          <MenuIcon />
        </IconButton>

        <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main', mr: 2 }}>
          SBTi Target Platform
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            Organization:
          </Typography>
          <Select
            size="small"
            value={orgId}
            onChange={handleOrgChange}
            sx={{ minWidth: 180, fontSize: '0.875rem' }}
          >
            <MenuItem value="org_default">GreenLang Corp</MenuItem>
            <MenuItem value="org_sub_1">EU Operations</MenuItem>
            <MenuItem value="org_sub_2">APAC Division</MenuItem>
          </Select>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            Reporting Year:
          </Typography>
          <Select
            size="small"
            value={reportingYear}
            onChange={handleYearChange}
            sx={{ minWidth: 100, fontSize: '0.875rem' }}
          >
            {YEARS.map((y) => (
              <MenuItem key={y} value={y}>{y}</MenuItem>
            ))}
          </Select>
        </Box>

        <Box sx={{ flexGrow: 1 }} />

        <Tooltip title="Export Report">
          <IconButton>
            <Download />
          </IconButton>
        </Tooltip>

        <Tooltip title="Help & Documentation">
          <IconButton>
            <HelpOutline />
          </IconButton>
        </Tooltip>

        <Tooltip title="Notifications">
          <IconButton>
            <Badge badgeContent={2} color="error">
              <Notifications />
            </Badge>
          </IconButton>
        </Tooltip>

        <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32, fontSize: '0.875rem' }}>
          GL
        </Avatar>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
