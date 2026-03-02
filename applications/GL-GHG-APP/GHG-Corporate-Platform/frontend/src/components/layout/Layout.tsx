/**
 * Layout - Main application shell
 *
 * Combines the Sidebar navigation, Header app bar, and content area
 * into the primary application layout. Accepts children prop for
 * rendering routed page content.
 */

import React from 'react';
import { Box, Toolbar } from '@mui/material';
import Sidebar, { DRAWER_WIDTH } from './Sidebar';
import Header from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f5f7f5' }}>
      <Sidebar />
      <Header />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          ml: `${DRAWER_WIDTH}px`,
          p: 3,
          minHeight: '100vh',
          maxWidth: `calc(100% - ${DRAWER_WIDTH}px)`,
          overflow: 'auto',
        }}
      >
        {/* Spacer to push content below fixed AppBar */}
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
