/**
 * Layout - Main application layout wrapper with sidebar, header, and content area.
 */

import React, { useState } from 'react';
import { Box, Toolbar } from '@mui/material';
import Sidebar from './Sidebar';
import Header from './Header';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { setActiveOrgId, selectActiveOrgId } from '../../store/slices/settingsSlice';

const DRAWER_WIDTH = 260;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [reportingYear, setReportingYear] = useState(2025);

  const handleToggleSidebar = () => {
    setSidebarOpen((prev) => !prev);
  };

  const handleOrgChange = (newOrgId: string) => {
    dispatch(setActiveOrgId(newOrgId));
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Header
        onToggleSidebar={handleToggleSidebar}
        orgId={orgId}
        onOrgChange={handleOrgChange}
        reportingYear={reportingYear}
        onYearChange={setReportingYear}
      />
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          ml: sidebarOpen ? `${DRAWER_WIDTH}px` : 0,
          transition: 'margin-left 225ms cubic-bezier(0.4, 0, 0.6, 1)',
          backgroundColor: '#F5F5F5',
          minHeight: '100vh',
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
