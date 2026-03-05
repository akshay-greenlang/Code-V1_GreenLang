/**
 * TimelineCountdown - Submission deadline countdown
 *
 * Displays a countdown to the CDP submission deadline with
 * color-coded urgency and date display.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { AccessTime, CalendarToday } from '@mui/icons-material';
import { formatDate } from '../../utils/formatters';

interface TimelineCountdownProps {
  deadline: string;
  daysRemaining: number;
}

const TimelineCountdown: React.FC<TimelineCountdownProps> = ({
  deadline,
  daysRemaining,
}) => {
  const getColor = (): string => {
    if (daysRemaining <= 0) return '#c62828';
    if (daysRemaining <= 7) return '#e53935';
    if (daysRemaining <= 30) return '#ef6c00';
    if (daysRemaining <= 60) return '#1565c0';
    return '#2e7d32';
  };

  const color = getColor();

  return (
    <Card sx={{ borderLeft: `4px solid ${color}` }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <AccessTime sx={{ color }} />
          <Typography variant="h6">Submission Deadline</Typography>
        </Box>
        <Typography variant="h3" fontWeight={800} sx={{ color }}>
          {daysRemaining > 0 ? daysRemaining : 0}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {daysRemaining > 0
            ? `days remaining`
            : daysRemaining === 0
            ? 'Due today'
            : `${Math.abs(daysRemaining)} days overdue`}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
          <CalendarToday fontSize="small" sx={{ color: 'text.secondary' }} />
          <Typography variant="caption" color="text.secondary">
            {formatDate(deadline)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default TimelineCountdown;
