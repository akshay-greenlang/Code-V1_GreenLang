/**
 * StatusBadge - Status indicator chip for disclosure statuses, action statuses, etc.
 */

import React from 'react';
import { Chip } from '@mui/material';

type StatusType =
  | 'not_started'
  | 'in_progress'
  | 'draft'
  | 'review'
  | 'final'
  | 'published'
  | 'completed'
  | 'overdue'
  | 'blocked'
  | 'open'
  | 'mitigating'
  | 'monitoring'
  | 'closed'
  | 'met'
  | 'partial'
  | 'not_met'
  | 'not_applicable';

interface StatusBadgeProps {
  status: string;
  size?: 'small' | 'medium';
}

const STATUS_CONFIG: Record<string, { label: string; color: 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' }> = {
  not_started: { label: 'Not Started', color: 'default' },
  in_progress: { label: 'In Progress', color: 'info' },
  draft: { label: 'Draft', color: 'warning' },
  review: { label: 'In Review', color: 'secondary' },
  final: { label: 'Final', color: 'primary' },
  published: { label: 'Published', color: 'success' },
  completed: { label: 'Completed', color: 'success' },
  overdue: { label: 'Overdue', color: 'error' },
  blocked: { label: 'Blocked', color: 'error' },
  open: { label: 'Open', color: 'info' },
  mitigating: { label: 'Mitigating', color: 'warning' },
  monitoring: { label: 'Monitoring', color: 'info' },
  closed: { label: 'Closed', color: 'default' },
  met: { label: 'Met', color: 'success' },
  partial: { label: 'Partial', color: 'warning' },
  not_met: { label: 'Not Met', color: 'error' },
  not_applicable: { label: 'N/A', color: 'default' },
  identified: { label: 'Identified', color: 'default' },
  evaluating: { label: 'Evaluating', color: 'info' },
  approved: { label: 'Approved', color: 'primary' },
  implementing: { label: 'Implementing', color: 'warning' },
  realized: { label: 'Realized', color: 'success' },
};

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'small' }) => {
  const config = STATUS_CONFIG[status] || { label: status.replace(/_/g, ' '), color: 'default' as const };

  return (
    <Chip
      label={config.label}
      color={config.color}
      size={size}
      variant="outlined"
      sx={{ fontWeight: 500, textTransform: 'capitalize' }}
    />
  );
};

export default StatusBadge;
