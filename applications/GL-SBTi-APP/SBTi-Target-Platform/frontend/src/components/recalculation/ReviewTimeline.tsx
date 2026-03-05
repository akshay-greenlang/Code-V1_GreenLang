/**
 * ReviewTimeline - Five-year review timeline.
 */
import React from 'react';
import { Card, CardContent, Typography, Stepper, Step, StepLabel, Box } from '@mui/material';
import type { FiveYearReview } from '../../types';
import { formatDate } from '../../utils/formatters';

interface ReviewTimelineProps { reviews: FiveYearReview[]; }

const ReviewTimeline: React.FC<ReviewTimelineProps> = ({ reviews }) => {
  const sorted = [...reviews].sort((a, b) => a.review_cycle - b.review_cycle);
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>Review Timeline</Typography>
        <Stepper alternativeLabel>
          {sorted.map((r) => (
            <Step key={r.id} completed={r.status === 'completed'} active={r.status === 'in_progress'}>
              <StepLabel>
                <Typography variant="body2" fontWeight={600}>Cycle {r.review_cycle}</Typography>
                <Typography variant="caption" color="text.secondary">Due: {formatDate(r.review_due_date)}</Typography>
                {r.outcome && <Typography variant="caption" display="block" color="text.secondary">{r.outcome.replace(/_/g, ' ')}</Typography>}
              </StepLabel>
            </Step>
          ))}
        </Stepper>
      </CardContent>
    </Card>
  );
};

export default ReviewTimeline;
