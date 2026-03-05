/**
 * LoadingSpinner - Centered loading indicator with optional message.
 */

import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface LoadingSpinnerProps {
  message?: string;
  size?: number;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ message = 'Loading...', size = 48 }) => (
  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: 2 }}>
    <CircularProgress size={size} />
    {message && <Typography variant="body2" color="text.secondary">{message}</Typography>}
  </Box>
);

export default LoadingSpinner;
