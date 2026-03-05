/**
 * CoverageTracker - Verification coverage progress tracker
 *
 * Displays coverage progress for each scope with targets,
 * showing how much of each emission scope is verified and
 * the gap to A-level requirements.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
} from '@mui/material';
import { TrackChanges } from '@mui/icons-material';
import type { VerificationRecord } from '../../types';

interface CoverageTrackerProps {
  records: VerificationRecord[];
}

interface ScopeCoverage {
  scope: string;
  totalCoverage: number;
  recordCount: number;
  target: number;
  gapToTarget: number;
}

const CoverageTracker: React.FC<CoverageTrackerProps> = ({ records }) => {
  // Calculate coverage by scope
  const scopeMap = new Map<string, VerificationRecord[]>();
  records.forEach((r) => {
    const existing = scopeMap.get(r.scope) || [];
    existing.push(r);
    scopeMap.set(r.scope, existing);
  });

  const SCOPE_TARGETS: Record<string, number> = {
    'Scope 1': 100,
    'Scope 2': 100,
    'Scope 3': 70,
  };

  const scopeCoverages: ScopeCoverage[] = ['Scope 1', 'Scope 2', 'Scope 3'].map(
    (scope) => {
      const scopeRecords = scopeMap.get(scope) || [];
      // Take the max coverage from records for this scope
      const totalCoverage = scopeRecords.length > 0
        ? Math.max(...scopeRecords.map((r) => r.coverage_pct))
        : 0;
      const target = SCOPE_TARGETS[scope] || 100;
      return {
        scope,
        totalCoverage,
        recordCount: scopeRecords.length,
        target,
        gapToTarget: Math.max(0, target - totalCoverage),
      };
    },
  );

  const overallCoverage = scopeCoverages.reduce(
    (sum, s) => sum + s.totalCoverage, 0,
  ) / 3;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <TrackChanges sx={{ color: '#1565c0' }} />
          <Typography variant="h6">Coverage Tracker</Typography>
          <Chip
            label={`Avg: ${overallCoverage.toFixed(0)}%`}
            size="small"
            color={overallCoverage >= 90 ? 'success' : 'primary'}
            sx={{ ml: 'auto' }}
          />
        </Box>

        {scopeCoverages.map((sc) => {
          const meetsTarget = sc.totalCoverage >= sc.target;
          const barColor = meetsTarget
            ? '#2e7d32'
            : sc.totalCoverage >= sc.target * 0.7
              ? '#ef6c00'
              : '#e53935';

          return (
            <Box key={sc.scope} sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" fontWeight={600}>
                    {sc.scope}
                  </Typography>
                  <Chip
                    label={`${sc.recordCount} record${sc.recordCount !== 1 ? 's' : ''}`}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: 10 }}
                  />
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" fontWeight={600}>
                    {sc.totalCoverage.toFixed(0)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    / {sc.target}% target
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ position: 'relative' }}>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(sc.totalCoverage, 100)}
                  sx={{
                    height: 10,
                    borderRadius: 5,
                    bgcolor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: barColor,
                      borderRadius: 5,
                    },
                  }}
                />
                {/* Target marker */}
                {sc.target < 100 && (
                  <Box
                    sx={{
                      position: 'absolute',
                      left: `${sc.target}%`,
                      top: -2,
                      width: 2,
                      height: 14,
                      bgcolor: '#455a64',
                    }}
                  />
                )}
              </Box>

              {sc.gapToTarget > 0 && (
                <Typography variant="caption" color="error">
                  {sc.gapToTarget.toFixed(0)}% gap to A-level target
                </Typography>
              )}
              {meetsTarget && (
                <Typography variant="caption" color="success.main">
                  Meets A-level requirement
                </Typography>
              )}
            </Box>
          );
        })}
      </CardContent>
    </Card>
  );
};

export default CoverageTracker;
