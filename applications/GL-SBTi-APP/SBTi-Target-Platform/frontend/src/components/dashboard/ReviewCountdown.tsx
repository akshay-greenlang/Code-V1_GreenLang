/**
 * ReviewCountdown - Five-year review countdown timer.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress } from '@mui/material';
import { Schedule } from '@mui/icons-material';
import { formatDate, formatDaysRemaining } from '../../utils/formatters';

interface ReviewCountdownProps {
  nextReviewDate: string;
  daysToReview: number;
}

const ReviewCountdown: React.FC<ReviewCountdownProps> = ({ nextReviewDate, daysToReview }) => {
  const totalDays = 365 * 5;
  const elapsed = totalDays - daysToReview;
  const pct = Math.min((elapsed / totalDays) * 100, 100);

  const isOverdue = daysToReview < 0;
  const isUrgent = daysToReview >= 0 && daysToReview <= 90;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Schedule sx={{ color: isOverdue ? 'error.main' : isUrgent ? 'warning.main' : 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Five-Year Review</Typography>
        </Box>
        <Typography variant="h3" sx={{
          fontWeight: 700,
          color: isOverdue ? 'error.main' : isUrgent ? 'warning.main' : 'primary.main',
          textAlign: 'center',
          mb: 1,
        }}>
          {Math.abs(daysToReview)}
        </Typography>
        <Typography variant="body2" sx={{ textAlign: 'center', mb: 2 }} color="text.secondary">
          {isOverdue ? 'days overdue' : 'days remaining'}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={pct}
          color={isOverdue ? 'error' : isUrgent ? 'warning' : 'primary'}
          sx={{ height: 6, borderRadius: 3, mb: 1 }}
        />
        <Typography variant="caption" color="text.secondary">
          Next review: {formatDate(nextReviewDate)}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default ReviewCountdown;
