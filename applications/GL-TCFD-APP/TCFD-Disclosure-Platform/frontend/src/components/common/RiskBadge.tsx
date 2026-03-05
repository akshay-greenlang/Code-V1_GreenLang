/**
 * RiskBadge - Risk level badge component with color coding from Very Low to Catastrophic.
 */

import React from 'react';
import { Chip } from '@mui/material';
import type { RiskLevel, ImpactSeverity, Likelihood } from '../../types';

type BadgeType = RiskLevel | ImpactSeverity | Likelihood | string;

interface RiskBadgeProps {
  level: BadgeType;
  size?: 'small' | 'medium';
  variant?: 'filled' | 'outlined';
}

const LEVEL_CONFIG: Record<string, { label: string; bgColor: string; textColor: string }> = {
  critical: { label: 'Critical', bgColor: '#B71C1C', textColor: '#FFFFFF' },
  catastrophic: { label: 'Catastrophic', bgColor: '#B71C1C', textColor: '#FFFFFF' },
  high: { label: 'High', bgColor: '#E65100', textColor: '#FFFFFF' },
  major: { label: 'Major', bgColor: '#E65100', textColor: '#FFFFFF' },
  almost_certain: { label: 'Almost Certain', bgColor: '#B71C1C', textColor: '#FFFFFF' },
  medium: { label: 'Medium', bgColor: '#F57F17', textColor: '#000000' },
  moderate: { label: 'Moderate', bgColor: '#F57F17', textColor: '#000000' },
  likely: { label: 'Likely', bgColor: '#E65100', textColor: '#FFFFFF' },
  possible: { label: 'Possible', bgColor: '#F57F17', textColor: '#000000' },
  low: { label: 'Low', bgColor: '#1B5E20', textColor: '#FFFFFF' },
  minor: { label: 'Minor', bgColor: '#2E7D32', textColor: '#FFFFFF' },
  unlikely: { label: 'Unlikely', bgColor: '#2E7D32', textColor: '#FFFFFF' },
  negligible: { label: 'Negligible', bgColor: '#388E3C', textColor: '#FFFFFF' },
  insignificant: { label: 'Insignificant', bgColor: '#388E3C', textColor: '#FFFFFF' },
  rare: { label: 'Rare', bgColor: '#388E3C', textColor: '#FFFFFF' },
  very_high: { label: 'Very High', bgColor: '#C62828', textColor: '#FFFFFF' },
  very_low: { label: 'Very Low', bgColor: '#4CAF50', textColor: '#FFFFFF' },
};

const RiskBadge: React.FC<RiskBadgeProps> = ({ level, size = 'small', variant = 'filled' }) => {
  const config = LEVEL_CONFIG[level] || {
    label: level.replace(/_/g, ' '),
    bgColor: '#9E9E9E',
    textColor: '#FFFFFF',
  };

  if (variant === 'outlined') {
    return (
      <Chip
        label={config.label}
        size={size}
        variant="outlined"
        sx={{
          borderColor: config.bgColor,
          color: config.bgColor,
          fontWeight: 600,
          textTransform: 'capitalize',
        }}
      />
    );
  }

  return (
    <Chip
      label={config.label}
      size={size}
      sx={{
        backgroundColor: config.bgColor,
        color: config.textColor,
        fontWeight: 600,
        textTransform: 'capitalize',
      }}
    />
  );
};

export default RiskBadge;
