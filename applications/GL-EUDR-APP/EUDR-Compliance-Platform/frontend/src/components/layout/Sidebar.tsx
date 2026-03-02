/**
 * Sidebar Navigation Component
 *
 * Collapsible MUI Drawer with EUDR-specific navigation items,
 * active state highlighting, and application branding.
 */

import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Typography,
  Box,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Business as BusinessIcon,
  Map as MapIcon,
  Warning as WarningIcon,
  Description as DescriptionIcon,
  Folder as FolderIcon,
  Timeline as TimelineIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  Forest as ForestIcon,
} from '@mui/icons-material';

const DRAWER_WIDTH_OPEN = 260;
const DRAWER_WIDTH_CLOSED = 72;

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactElement;
}

const navItems: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { label: 'Suppliers', path: '/suppliers', icon: <BusinessIcon /> },
  { label: 'Plot Registry', path: '/plots', icon: <MapIcon /> },
  { label: 'Risk Assessment', path: '/risk', icon: <WarningIcon /> },
  { label: 'DDS Management', path: '/dds', icon: <DescriptionIcon /> },
  { label: 'Documents', path: '/documents', icon: <FolderIcon /> },
  { label: 'Pipeline', path: '/pipeline', icon: <TimelineIcon /> },
];

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open, onToggle }) => {
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
        width: open ? DRAWER_WIDTH_OPEN : DRAWER_WIDTH_CLOSED,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? DRAWER_WIDTH_OPEN : DRAWER_WIDTH_CLOSED,
          boxSizing: 'border-box',
          transition: 'width 0.2s ease-in-out',
          overflowX: 'hidden',
          backgroundColor: '#1b3a1b',
          color: '#ffffff',
        },
      }}
    >
      {/* Branding */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          p: 2,
          minHeight: 64,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ForestIcon sx={{ color: '#66bb6a', fontSize: 32 }} />
          {open && (
            <Typography
              variant="h6"
              noWrap
              sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}
            >
              GL-EUDR
            </Typography>
          )}
        </Box>
        <IconButton onClick={onToggle} sx={{ color: '#ffffff' }}>
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255,255,255,0.12)' }} />

      {/* Navigation Items */}
      <List sx={{ px: 1, pt: 1 }}>
        {navItems.map((item) => {
          const active = isActive(item.path);
          return (
            <Tooltip
              key={item.path}
              title={open ? '' : item.label}
              placement="right"
              arrow
            >
              <ListItem disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  onClick={() => navigate(item.path)}
                  sx={{
                    minHeight: 48,
                    justifyContent: open ? 'initial' : 'center',
                    borderRadius: 1,
                    backgroundColor: active
                      ? 'rgba(102, 187, 106, 0.2)'
                      : 'transparent',
                    '&:hover': {
                      backgroundColor: active
                        ? 'rgba(102, 187, 106, 0.3)'
                        : 'rgba(255, 255, 255, 0.08)',
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: open ? 2 : 'auto',
                      justifyContent: 'center',
                      color: active ? '#66bb6a' : 'rgba(255,255,255,0.7)',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  {open && (
                    <ListItemText
                      primary={item.label}
                      primaryTypographyProps={{
                        fontSize: '0.9rem',
                        fontWeight: active ? 600 : 400,
                        color: active ? '#ffffff' : 'rgba(255,255,255,0.7)',
                      }}
                    />
                  )}
                </ListItemButton>
              </ListItem>
            </Tooltip>
          );
        })}
      </List>

      {/* Footer */}
      {open && (
        <Box sx={{ mt: 'auto', p: 2, textAlign: 'center' }}>
          <Typography
            variant="caption"
            sx={{ color: 'rgba(255,255,255,0.4)' }}
          >
            EU Reg. 2023/1115
          </Typography>
        </Box>
      )}
    </Drawer>
  );
};

export default Sidebar;
