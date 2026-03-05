/**
 * DeforestationTracker - Deforestation commitment status and timeline.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Alert } from '@mui/material';
import { CheckCircle, Warning } from '@mui/icons-material';
import { formatDate } from '../../utils/formatters';

interface DeforestationTrackerProps { hasCommitment: boolean; commitmentDate: string | null; zeroByYear: number | null; }

const DeforestationTracker: React.FC<DeforestationTrackerProps> = ({ hasCommitment, commitmentDate, zeroByYear }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Deforestation Commitment</Typography>
      {hasCommitment ? (
        <Box>
          <Alert severity="success" icon={<CheckCircle />} sx={{ mb: 2 }}>
            Zero deforestation commitment is in place.
          </Alert>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {commitmentDate && <Chip label={`Committed: ${formatDate(commitmentDate)}`} size="small" />}
            {zeroByYear && <Chip label={`Zero by: ${zeroByYear}`} size="small" color="primary" />}
          </Box>
        </Box>
      ) : (
        <Alert severity="warning" icon={<Warning />}>
          No zero deforestation commitment detected. SBTi FLAG guidance requires companies to commit to zero deforestation by 2025.
        </Alert>
      )}
    </CardContent>
  </Card>
);

export default DeforestationTracker;
