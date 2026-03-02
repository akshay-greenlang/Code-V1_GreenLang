/**
 * StatusBadge - Reusable status indicator chip
 *
 * Displays a color-coded badge for various statuses:
 *   - Verification status (not_started, in_progress, approved, rejected)
 *   - Data quality tier (tier_1 through tier_4)
 *   - Scope badges (scope_1, scope_2, scope_3)
 *   - Inventory status (draft, in_review, approved, verified, published)
 *   - Report status (draft, review, final, published)
 *   - Target status (on_track, at_risk, off_track, achieved)
 *   - Finding severity (low, medium, high, critical)
 *   - Materiality (material, immaterial, not_calculated)
 *   - General (active, expired, pending, generating, ready, failed)
 */

import React from 'react';
import { Chip, ChipProps } from '@mui/material';

interface StatusBadgeProps {
  status: string;
  size?: 'small' | 'medium';
  variant?: 'filled' | 'outlined';
}

interface StatusConfig {
  label: string;
  color: ChipProps['color'];
  sx?: Record<string, unknown>;
}

const STATUS_CONFIG: Record<string, StatusConfig> = {
  // Verification status
  not_started: { label: 'Not Started', color: 'default' },
  in_progress: { label: 'In Progress', color: 'info' },
  pending_review: { label: 'Pending Review', color: 'warning' },
  approved: { label: 'Approved', color: 'success' },
  rejected: { label: 'Rejected', color: 'error' },

  // Inventory status
  draft: { label: 'Draft', color: 'default' },
  in_review: { label: 'In Review', color: 'warning' },
  verified: { label: 'Verified', color: 'success' },
  published: { label: 'Published', color: 'success' },

  // Report status
  review: { label: 'Review', color: 'warning' },
  final: { label: 'Final', color: 'info' },

  // Target status
  on_track: { label: 'On Track', color: 'success' },
  at_risk: { label: 'At Risk', color: 'warning' },
  off_track: { label: 'Off Track', color: 'error' },
  achieved: { label: 'Achieved', color: 'success' },

  // Data quality tiers
  tier_1: { label: 'Tier 1', color: 'success' },
  tier_2: { label: 'Tier 2', color: 'info' },
  tier_3: { label: 'Tier 3', color: 'warning' },
  tier_4: { label: 'Tier 4', color: 'error' },

  // Scope badges
  scope_1: { label: 'Scope 1', color: 'default', sx: { backgroundColor: '#e53935', color: '#fff' } },
  scope_2: { label: 'Scope 2', color: 'default', sx: { backgroundColor: '#1e88e5', color: '#fff' } },
  scope_3: { label: 'Scope 3', color: 'default', sx: { backgroundColor: '#43a047', color: '#fff' } },

  // Materiality
  material: { label: 'Material', color: 'success' },
  immaterial: { label: 'Immaterial', color: 'default' },
  not_calculated: { label: 'Not Calculated', color: 'warning' },

  // Verification level
  limited: { label: 'Limited', color: 'warning' },
  reasonable: { label: 'Reasonable', color: 'success' },
  not_verified: { label: 'Not Verified', color: 'default' },

  // Finding severity
  low: { label: 'Low', color: 'info' },
  medium: { label: 'Medium', color: 'warning' },
  high: { label: 'High', color: 'error' },
  critical: { label: 'Critical', color: 'error' },

  // Finding status
  open: { label: 'Open', color: 'error' },
  resolved: { label: 'Resolved', color: 'success' },
  accepted: { label: 'Accepted', color: 'info' },

  // General
  active: { label: 'Active', color: 'success' },
  expired: { label: 'Expired', color: 'default' },
  pending: { label: 'Pending', color: 'warning' },
  generating: { label: 'Generating', color: 'info' },
  ready: { label: 'Ready', color: 'success' },
  failed: { label: 'Failed', color: 'error' },

  // SBTi
  aligned: { label: 'SBTi Aligned', color: 'success' },
  not_aligned: { label: 'Not Aligned', color: 'error' },
};

const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'small',
  variant = 'filled',
}) => {
  const config = STATUS_CONFIG[status] || {
    label: status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    color: 'default' as const,
  };

  return (
    <Chip
      label={config.label}
      color={config.color}
      size={size}
      variant={variant}
      sx={{ fontWeight: 500, ...config.sx }}
    />
  );
};

export default StatusBadge;
