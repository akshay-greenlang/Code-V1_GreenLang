/**
 * DashboardPage - Executive overview of GHG inventory
 * Placeholder component; full implementation in a subsequent build phase.
 */

import React from 'react';
import { Typography, Box } from '@mui/material';

const DashboardPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        GHG Inventory Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary">
        Executive overview of corporate greenhouse gas emissions across all scopes.
      </Typography>
    </Box>
  );
};

export default DashboardPage;
