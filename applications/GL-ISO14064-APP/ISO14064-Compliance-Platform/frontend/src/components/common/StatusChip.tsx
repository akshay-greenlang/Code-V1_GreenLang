/**
 * StatusChip - Colored chip for status values
 *
 * Renders a MUI Chip colored according to the status value
 * (draft, verified, open, resolved, etc.).  Uses the getStatusColor
 * utility for consistent color mapping.
 */

import React from 'react';
import { Chip, ChipProps } from '@mui/material';
import { getStatusColor } from '../../utils/formatters';

interface StatusChipProps {
  status: string;
  label?: string;
  size?: ChipProps['size'];
  variant?: ChipProps['variant'];
}

const StatusChip: React.FC<StatusChipProps> = ({
  status,
  label,
  size = 'small',
  variant = 'filled',
}) => {
  const displayLabel = label || status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const color = getStatusColor(status);

  return (
    <Chip
      label={displayLabel}
      color={color}
      size={size}
      variant={variant}
      sx={{ fontWeight: 500 }}
    />
  );
};

export default StatusChip;
