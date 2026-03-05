/**
 * RAGIndicator - Red/Amber/Green on-track indicator with details.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { CheckCircle, Warning, Cancel } from '@mui/icons-material';
import type { RAGStatus } from '../../types';
import { getRAGColor } from '../../utils/pathwayHelpers';
import { formatRAGStatus } from '../../utils/formatters';

interface RAGIndicatorProps { status: RAGStatus; progressPct: number; gapToTarget: number; }

const RAGIndicator: React.FC<RAGIndicatorProps> = ({ status, progressPct, gapToTarget }) => {
  const color = getRAGColor(status);
  const icon = status === 'on_track' ? <CheckCircle sx={{ fontSize: 48, color }} /> : status === 'at_risk' ? <Warning sx={{ fontSize: 48, color }} /> : <Cancel sx={{ fontSize: 48, color }} />;
  return (
    <Card>
      <CardContent sx={{ textAlign: 'center' }}>
        {icon}
        <Typography variant="h5" sx={{ fontWeight: 700, color, mt: 1 }}>{formatRAGStatus(status)}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Progress: {progressPct.toFixed(1)}%</Typography>
        <Typography variant="body2" color="text.secondary">Gap to target: {gapToTarget.toLocaleString()} tCO2e</Typography>
      </CardContent>
    </Card>
  );
};

export default RAGIndicator;
