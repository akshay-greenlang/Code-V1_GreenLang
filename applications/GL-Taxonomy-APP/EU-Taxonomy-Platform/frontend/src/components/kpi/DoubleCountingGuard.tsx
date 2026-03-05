/**
 * DoubleCountingGuard - Indicator showing double-counting prevention status.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Alert, Chip } from '@mui/material';
import { VerifiedUser, Warning } from '@mui/icons-material';

interface DoubleCountingGuardProps {
  hasDuplicates?: boolean;
  activitiesChecked?: number;
  issuesFound?: number;
}

const DoubleCountingGuard: React.FC<DoubleCountingGuardProps> = ({
  hasDuplicates = false,
  activitiesChecked = 38,
  issuesFound = 0,
}) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        {hasDuplicates ? <Warning color="error" /> : <VerifiedUser color="success" />}
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Double-Counting Prevention
        </Typography>
      </Box>

      {hasDuplicates ? (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Potential double-counting detected in {issuesFound} activities. Review required before finalizing KPIs.
        </Alert>
      ) : (
        <Alert severity="success" sx={{ mb: 2 }}>
          No double-counting detected. Each activity is assigned to a single primary objective.
        </Alert>
      )}

      <Box sx={{ display: 'flex', gap: 2 }}>
        <Chip label={`${activitiesChecked} activities checked`} variant="outlined" size="small" />
        <Chip label={`${issuesFound} issues found`} color={issuesFound > 0 ? 'error' : 'success'} size="small" />
      </Box>
    </CardContent>
  </Card>
);

export default DoubleCountingGuard;
