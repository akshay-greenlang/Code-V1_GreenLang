/**
 * EngagementDashboard - Supplier engagement metrics
 */
import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress } from '@mui/material';
import { People, CheckCircle, Send, Assessment } from '@mui/icons-material';
import type { SupplyChainSummary } from '../../types';

interface EngagementDashboardProps { summary: SupplyChainSummary; }

const EngagementDashboard: React.FC<EngagementDashboardProps> = ({ summary }) => {
  const metrics = [
    { label: 'Total Suppliers', value: summary.total_suppliers, icon: <People sx={{ color: '#1565c0' }} /> },
    { label: 'Invited', value: summary.invited_count, icon: <Send sx={{ color: '#ef6c00' }} /> },
    { label: 'Responded', value: summary.responded_count, icon: <CheckCircle sx={{ color: '#2e7d32' }} /> },
    { label: 'Scored', value: summary.scored_count, icon: <Assessment sx={{ color: '#7b1fa2' }} /> },
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Engagement Overview</Typography>
        <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
          {metrics.map((m) => (
            <Box key={m.label} sx={{ textAlign: 'center', flex: 1 }}>
              {m.icon}
              <Typography variant="h5" fontWeight={700}>{m.value}</Typography>
              <Typography variant="caption" color="text.secondary">{m.label}</Typography>
            </Box>
          ))}
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>Response Rate</Typography>
        <LinearProgress variant="determinate" value={summary.response_rate} sx={{ height: 8, borderRadius: 4, mb: 0.5, '& .MuiLinearProgress-bar': { backgroundColor: '#1b5e20' } }} />
        <Typography variant="caption" color="text.secondary">{summary.response_rate.toFixed(0)}%</Typography>
      </CardContent>
    </Card>
  );
};

export default EngagementDashboard;
