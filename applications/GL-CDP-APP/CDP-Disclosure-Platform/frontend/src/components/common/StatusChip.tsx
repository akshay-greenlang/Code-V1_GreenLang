/**
 * StatusChip - Response status chip
 *
 * Displays a colored chip for CDP response statuses:
 * not_started, draft, in_review, approved, submitted.
 */

import React from 'react';
import { Chip } from '@mui/material';
import { getStatusColor } from '../../utils/formatters';

interface StatusChipProps {
  status: string;
  size?: 'small' | 'medium';
}

const STATUS_LABELS: Record<string, string> = {
  not_started: 'Not Started',
  draft: 'Draft',
  in_review: 'In Review',
  approved: 'Approved',
  submitted: 'Submitted',
  invited: 'Invited',
  in_progress: 'In Progress',
  scored: 'Scored',
  declined: 'Declined',
  completed: 'Completed',
  delayed: 'Delayed',
  limited: 'Limited',
  reasonable: 'Reasonable',
  not_verified: 'Not Verified',
};

const StatusChip: React.FC<StatusChipProps> = ({ status, size = 'small' }) => {
  const label = STATUS_LABELS[status] || status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <Chip
      label={label}
      color={getStatusColor(status)}
      size={size}
      variant="outlined"
    />
  );
};

export default StatusChip;
