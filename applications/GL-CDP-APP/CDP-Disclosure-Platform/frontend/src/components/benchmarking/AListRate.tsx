/**
 * AListRate - A-list achievement rate visualization
 */
import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import { EmojiEvents } from '@mui/icons-material';

interface AListRateProps { aListCount: number; totalCompanies: number; aListRate: number; }

const AListRate: React.FC<AListRateProps> = ({ aListCount, totalCompanies, aListRate }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <EmojiEvents sx={{ color: '#fbc02d' }} />
        <Typography variant="h6">A-List Rate</Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <CircularProgress variant="determinate" value={aListRate} size={80} thickness={4} sx={{ color: '#1b5e20' }} />
          <Box sx={{ position: 'absolute', top: 0, left: 0, bottom: 0, right: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="h6" fontWeight={700} color="primary">{aListRate.toFixed(0)}%</Typography>
          </Box>
        </Box>
        <Box>
          <Typography variant="body2">{aListCount} companies achieved A/A-</Typography>
          <Typography variant="caption" color="text.secondary">Out of {totalCompanies} in sector</Typography>
        </Box>
      </Box>
    </CardContent>
  </Card>
);

export default AListRate;
