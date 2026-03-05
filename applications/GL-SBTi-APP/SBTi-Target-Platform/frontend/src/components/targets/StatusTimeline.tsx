/**
 * StatusTimeline - Target status lifecycle visualization.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Stepper, Step, StepLabel, StepConnector } from '@mui/material';
import type { TargetStatus } from '../../types';

interface StatusTimelineProps {
  currentStatus: TargetStatus;
  timeline?: { date: string; status: string; notes: string }[];
}

const STATUS_STEPS: { status: TargetStatus; label: string }[] = [
  { status: 'draft', label: 'Draft' },
  { status: 'submitted', label: 'Submitted' },
  { status: 'under_review', label: 'Under Review' },
  { status: 'approved', label: 'Approved' },
  { status: 'validated', label: 'Validated' },
];

const StatusTimeline: React.FC<StatusTimelineProps> = ({ currentStatus }) => {
  const activeStep = STATUS_STEPS.findIndex((s) => s.status === currentStatus);
  const isRejected = currentStatus === 'rejected';
  const isWithdrawn = currentStatus === 'withdrawn';

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>Target Lifecycle</Typography>
        {isRejected || isWithdrawn ? (
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Typography variant="h6" color={isRejected ? 'error' : 'text.secondary'}>
              Target {isRejected ? 'Rejected' : 'Withdrawn'}
            </Typography>
          </Box>
        ) : (
          <Stepper activeStep={activeStep} alternativeLabel>
            {STATUS_STEPS.map((step) => (
              <Step key={step.status}>
                <StepLabel>{step.label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        )}
      </CardContent>
    </Card>
  );
};

export default StatusTimeline;
