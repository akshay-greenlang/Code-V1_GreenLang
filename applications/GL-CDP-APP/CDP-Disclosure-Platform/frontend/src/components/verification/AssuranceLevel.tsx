/**
 * AssuranceLevel - Assurance level comparison widget
 *
 * Visually compares limited vs reasonable assurance across scopes,
 * showing which level of assurance is applied and CDP scoring
 * implications.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import { Security, ArrowForward } from '@mui/icons-material';
import type { VerificationRecord } from '../../types';
import { VerificationLevel } from '../../types';

interface AssuranceLevelProps {
  records: VerificationRecord[];
}

const LEVEL_CONFIG: Record<
  VerificationLevel,
  { label: string; color: string; chipColor: 'success' | 'warning' | 'default'; description: string }
> = {
  [VerificationLevel.REASONABLE]: {
    label: 'Reasonable',
    color: '#1b5e20',
    chipColor: 'success',
    description: 'Highest level of assurance; required for A-level scoring in Scope 1 & 2',
  },
  [VerificationLevel.LIMITED]: {
    label: 'Limited',
    color: '#ef6c00',
    chipColor: 'warning',
    description: 'Moderate assurance; acceptable for B-level scoring and Scope 3',
  },
  [VerificationLevel.NOT_VERIFIED]: {
    label: 'Not Verified',
    color: '#9e9e9e',
    chipColor: 'default',
    description: 'No third-party verification; impacts scoring negatively',
  },
};

const AssuranceLevel: React.FC<AssuranceLevelProps> = ({ records }) => {
  // Group records by scope and determine highest assurance per scope
  const scopeAssurance = new Map<string, VerificationLevel>();
  const LEVEL_PRIORITY: Record<VerificationLevel, number> = {
    [VerificationLevel.REASONABLE]: 3,
    [VerificationLevel.LIMITED]: 2,
    [VerificationLevel.NOT_VERIFIED]: 1,
  };

  records.forEach((r) => {
    const current = scopeAssurance.get(r.scope);
    if (!current || LEVEL_PRIORITY[r.verification_level] > LEVEL_PRIORITY[current]) {
      scopeAssurance.set(r.scope, r.verification_level);
    }
  });

  const scopes = ['Scope 1', 'Scope 2', 'Scope 3'];
  const reasonableCount = Array.from(scopeAssurance.values()).filter(
    (v) => v === VerificationLevel.REASONABLE,
  ).length;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Security sx={{ color: '#1565c0' }} />
          <Typography variant="h6">Assurance Levels</Typography>
          <Chip
            label={`${reasonableCount}/3 reasonable`}
            size="small"
            color={reasonableCount >= 2 ? 'success' : reasonableCount >= 1 ? 'warning' : 'default'}
            sx={{ ml: 'auto' }}
          />
        </Box>

        {/* Scope assurance display */}
        {scopes.map((scope) => {
          const level = scopeAssurance.get(scope) || VerificationLevel.NOT_VERIFIED;
          const config = LEVEL_CONFIG[level];

          return (
            <Box
              key={scope}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                py: 1.5,
                borderBottom: '1px solid #f0f0f0',
              }}
            >
              <Typography variant="body2" fontWeight={600} sx={{ minWidth: 70 }}>
                {scope}
              </Typography>

              {/* Level indicator bar */}
              <Box sx={{ display: 'flex', gap: 0.5, flex: 1 }}>
                {Object.values(VerificationLevel).map((vl) => {
                  const cfg = LEVEL_CONFIG[vl];
                  const isActive = vl === level;
                  return (
                    <Box
                      key={vl}
                      sx={{
                        flex: 1,
                        height: 8,
                        borderRadius: 4,
                        bgcolor: isActive ? cfg.color : '#e0e0e0',
                        opacity: isActive ? 1 : 0.3,
                        transition: 'all 0.3s',
                      }}
                    />
                  );
                })}
              </Box>

              <Chip
                label={config.label}
                size="small"
                color={config.chipColor}
                sx={{ minWidth: 90, fontSize: 11 }}
              />
            </Box>
          );
        })}

        {/* Upgrade path */}
        <Box sx={{ mt: 2, p: 1.5, bgcolor: '#f5f7f5', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Scoring Impact
          </Typography>
          {Object.entries(LEVEL_CONFIG).map(([level, config]) => (
            <Box
              key={level}
              sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}
            >
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: config.color,
                  flexShrink: 0,
                }}
              />
              <Typography variant="caption" color="text.secondary">
                <strong>{config.label}:</strong> {config.description}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Upgrade suggestion */}
        {scopeAssurance.has('Scope 1') &&
         scopeAssurance.get('Scope 1') === VerificationLevel.LIMITED && (
          <Box
            sx={{
              mt: 2,
              p: 1.5,
              bgcolor: '#fff3e0',
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <ArrowForward sx={{ color: '#ef6c00', fontSize: 18 }} />
            <Typography variant="body2" color="warning.dark">
              Upgrading Scope 1 from Limited to Reasonable assurance is required for A-level scoring.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AssuranceLevel;
