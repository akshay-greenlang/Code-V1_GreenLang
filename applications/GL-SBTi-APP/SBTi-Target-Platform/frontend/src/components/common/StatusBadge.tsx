/**
 * StatusBadge - RAG status badge (on-track, off-track, at-risk) and general status chips.
 */

import React from 'react';
import { Chip } from '@mui/material';
import type { RAGStatus, TargetStatus, PathwayAlignment } from '../../types';

interface StatusBadgeProps {
  status: string;
  variant?: 'rag' | 'target' | 'alignment' | 'validation';
  size?: 'small' | 'medium';
}

const RAG_COLORS: Record<RAGStatus, { bg: string; color: string }> = {
  on_track: { bg: '#E8F5E9', color: '#1B5E20' },
  at_risk: { bg: '#FFF3E0', color: '#E65100' },
  off_track: { bg: '#FFEBEE', color: '#B71C1C' },
};

const TARGET_COLORS: Record<string, { bg: string; color: string }> = {
  validated: { bg: '#E8F5E9', color: '#1B5E20' },
  approved: { bg: '#E8F5E9', color: '#2E7D32' },
  committed: { bg: '#E3F2FD', color: '#0D47A1' },
  submitted: { bg: '#FFF3E0', color: '#E65100' },
  under_review: { bg: '#FFF3E0', color: '#EF6C00' },
  draft: { bg: '#F5F5F5', color: '#616161' },
  rejected: { bg: '#FFEBEE', color: '#B71C1C' },
  expired: { bg: '#FFEBEE', color: '#C62828' },
  withdrawn: { bg: '#ECEFF1', color: '#455A64' },
};

const ALIGNMENT_COLORS: Record<string, { bg: string; color: string }> = {
  '1.5C': { bg: '#E8F5E9', color: '#1B5E20' },
  well_below_2C: { bg: '#E8F5E9', color: '#2E7D32' },
  '2C': { bg: '#FFF3E0', color: '#E65100' },
  above_2C: { bg: '#FFEBEE', color: '#B71C1C' },
  not_aligned: { bg: '#FFEBEE', color: '#C62828' },
};

const VALIDATION_COLORS: Record<string, { bg: string; color: string }> = {
  pass: { bg: '#E8F5E9', color: '#1B5E20' },
  fail: { bg: '#FFEBEE', color: '#B71C1C' },
  warning: { bg: '#FFF3E0', color: '#E65100' },
  not_applicable: { bg: '#F5F5F5', color: '#9E9E9E' },
};

function getColors(status: string, variant: string): { bg: string; color: string } {
  switch (variant) {
    case 'rag': return RAG_COLORS[status as RAGStatus] || { bg: '#F5F5F5', color: '#616161' };
    case 'target': return TARGET_COLORS[status] || { bg: '#F5F5F5', color: '#616161' };
    case 'alignment': return ALIGNMENT_COLORS[status] || { bg: '#F5F5F5', color: '#616161' };
    case 'validation': return VALIDATION_COLORS[status] || { bg: '#F5F5F5', color: '#616161' };
    default: return { bg: '#F5F5F5', color: '#616161' };
  }
}

function formatLabel(status: string): string {
  if (status === '1.5C') return '1.5\u00B0C';
  if (status === '2C') return '2\u00B0C';
  if (status === 'well_below_2C') return 'WB2\u00B0C';
  return status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, variant = 'rag', size = 'small' }) => {
  const colors = getColors(status, variant);
  return (
    <Chip
      label={formatLabel(status)}
      size={size}
      sx={{
        backgroundColor: colors.bg,
        color: colors.color,
        fontWeight: 600,
        fontSize: size === 'small' ? '0.75rem' : '0.8125rem',
      }}
    />
  );
};

export default StatusBadge;
