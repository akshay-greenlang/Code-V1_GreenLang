/**
 * VerificationStatus - Overall verification status overview
 *
 * Displays the verification status for Scope 1, 2, and 3 emissions
 * with coverage levels, A-level eligibility indicators, and summary
 * chips for quick status assessment.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  VerifiedUser,
  CheckCircle,
  Cancel,
  Shield,
} from '@mui/icons-material';
import type { VerificationSummary } from '../../types';
import { VerificationLevel } from '../../types';

interface VerificationStatusProps {
  summary: VerificationSummary;
}

function getLevelLabel(level: VerificationLevel): string {
  switch (level) {
    case VerificationLevel.REASONABLE: return 'Reasonable';
    case VerificationLevel.LIMITED: return 'Limited';
    case VerificationLevel.NOT_VERIFIED: return 'Not Verified';
  }
}

function getLevelColor(level: VerificationLevel): string {
  switch (level) {
    case VerificationLevel.REASONABLE: return '#1b5e20';
    case VerificationLevel.LIMITED: return '#ef6c00';
    case VerificationLevel.NOT_VERIFIED: return '#9e9e9e';
  }
}

interface ScopeCardProps {
  label: string;
  verified: boolean;
  coverage: number;
  level: VerificationLevel;
  aLevelTarget?: string;
  meetsALevel?: boolean;
}

const ScopeCard: React.FC<ScopeCardProps> = ({
  label,
  verified,
  coverage,
  level,
  aLevelTarget,
  meetsALevel,
}) => (
  <Box
    sx={{
      flex: 1,
      p: 2,
      borderRadius: 1,
      border: '1px solid',
      borderColor: verified ? '#c8e6c9' : '#e0e0e0',
      bgcolor: verified ? '#f1f8e9' : '#fafafa',
    }}
  >
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
      {verified ? (
        <VerifiedUser sx={{ color: '#2e7d32', fontSize: 22 }} />
      ) : (
        <Shield sx={{ color: '#9e9e9e', fontSize: 22 }} />
      )}
      <Typography variant="subtitle1" fontWeight={600}>
        {label}
      </Typography>
    </Box>

    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          Coverage
        </Typography>
        <Typography variant="caption" fontWeight={600}>
          {coverage.toFixed(0)}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={coverage}
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: '#e0e0e0',
          '& .MuiLinearProgress-bar': {
            bgcolor: getLevelColor(level),
            borderRadius: 3,
          },
        }}
      />
    </Box>

    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
      <Chip
        label={getLevelLabel(level)}
        size="small"
        sx={{
          fontSize: 10,
          bgcolor: getLevelColor(level) + '20',
          color: getLevelColor(level),
          border: `1px solid ${getLevelColor(level)}40`,
        }}
      />
      {aLevelTarget && (
        <Chip
          icon={meetsALevel
            ? <CheckCircle sx={{ fontSize: 12 }} />
            : <Cancel sx={{ fontSize: 12 }} />}
          label={aLevelTarget}
          size="small"
          color={meetsALevel ? 'success' : 'default'}
          variant="outlined"
          sx={{ fontSize: 10 }}
        />
      )}
    </Box>
  </Box>
);

const VerificationStatus: React.FC<VerificationStatusProps> = ({ summary }) => {
  const allVerified = summary.scope_1_verified
    && summary.scope_2_verified
    && summary.scope_3_verified;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">Verification Status</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={summary.overall_meets_a_level ? 'A-Level Ready' : 'Not A-Level Ready'}
              size="small"
              color={summary.overall_meets_a_level ? 'success' : 'warning'}
            />
            {allVerified && (
              <Chip label="All Scopes Verified" size="small" color="success" variant="outlined" />
            )}
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <ScopeCard
            label="Scope 1"
            verified={summary.scope_1_verified}
            coverage={summary.scope_1_coverage}
            level={summary.scope_1_level}
            aLevelTarget="100% required"
            meetsALevel={summary.meets_a_level_scope12 && summary.scope_1_coverage >= 100}
          />
          <ScopeCard
            label="Scope 2"
            verified={summary.scope_2_verified}
            coverage={summary.scope_2_coverage}
            level={summary.scope_2_level}
            aLevelTarget="100% required"
            meetsALevel={summary.meets_a_level_scope12 && summary.scope_2_coverage >= 100}
          />
          <ScopeCard
            label="Scope 3"
            verified={summary.scope_3_verified}
            coverage={summary.scope_3_coverage}
            level={summary.scope_3_level}
            aLevelTarget=">= 70% 1 cat"
            meetsALevel={summary.meets_a_level_scope3}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default VerificationStatus;
