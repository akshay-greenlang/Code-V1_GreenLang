/**
 * StatusBadge - Colored chip for displaying alignment/assessment status.
 */

import React from 'react';
import { Chip } from '@mui/material';
import { AlignmentStatus, DNSHStatus } from '../../types';
import { alignmentStatusLabel, dnshStatusLabel, dnshStatusColor } from '../../utils/taxonomyHelpers';
import { alignmentStatusColor } from '../../utils/formatters';

interface StatusBadgeProps {
  status: AlignmentStatus | DNSHStatus | string;
  type?: 'alignment' | 'dnsh' | 'custom';
  size?: 'small' | 'medium';
  customColor?: string;
  customLabel?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  type = 'alignment',
  size = 'small',
  customColor,
  customLabel,
}) => {
  const getLabel = () => {
    if (customLabel) return customLabel;
    if (type === 'alignment') return alignmentStatusLabel(status as AlignmentStatus);
    if (type === 'dnsh') return dnshStatusLabel(status as DNSHStatus);
    return status;
  };

  const getColor = () => {
    if (customColor) return customColor;
    if (type === 'alignment') return alignmentStatusColor(status);
    if (type === 'dnsh') return dnshStatusColor(status as DNSHStatus);
    return '#757575';
  };

  return (
    <Chip
      label={getLabel()}
      size={size}
      sx={{
        backgroundColor: getColor(),
        color: '#FFFFFF',
        fontWeight: 600,
        fontSize: size === 'small' ? '0.75rem' : '0.875rem',
      }}
    />
  );
};

export default StatusBadge;
