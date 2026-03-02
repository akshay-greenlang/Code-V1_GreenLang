/**
 * StatusBadge Component
 *
 * Color-coded MUI Chip for displaying compliance status, risk level,
 * DDS status, and pipeline stage values with consistent styling.
 */

import React from 'react';
import { Chip, ChipProps } from '@mui/material';
import type {
  ComplianceStatus,
  RiskLevel,
  DDSStatus,
  PipelineStage,
} from '../../types';

type StatusValue = ComplianceStatus | RiskLevel | DDSStatus | string;

interface StatusBadgeProps {
  status: StatusValue;
  size?: ChipProps['size'];
}

const statusConfig: Record<
  string,
  { label: string; color: ChipProps['color'] }
> = {
  // ComplianceStatus
  compliant: { label: 'Compliant', color: 'success' },
  non_compliant: { label: 'Non-Compliant', color: 'error' },
  pending: { label: 'Pending', color: 'warning' },
  under_review: { label: 'Under Review', color: 'info' },
  expired: { label: 'Expired', color: 'default' },

  // RiskLevel
  low: { label: 'Low', color: 'success' },
  standard: { label: 'Standard', color: 'info' },
  high: { label: 'High', color: 'warning' },
  critical: { label: 'Critical', color: 'error' },

  // DDSStatus
  draft: { label: 'Draft', color: 'default' },
  pending_review: { label: 'Pending Review', color: 'warning' },
  validated: { label: 'Validated', color: 'info' },
  submitted: { label: 'Submitted', color: 'primary' },
  accepted: { label: 'Accepted', color: 'success' },
  rejected: { label: 'Rejected', color: 'error' },
  amended: { label: 'Amended', color: 'warning' },

  // Pipeline status
  running: { label: 'Running', color: 'info' },
  completed: { label: 'Completed', color: 'success' },
  failed: { label: 'Failed', color: 'error' },
  cancelled: { label: 'Cancelled', color: 'default' },
  skipped: { label: 'Skipped', color: 'default' },

  // Document verification
  verified: { label: 'Verified', color: 'success' },
};

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'small' }) => {
  const config = statusConfig[status] || {
    label: status.replace(/_/g, ' '),
    color: 'default' as const,
  };

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
