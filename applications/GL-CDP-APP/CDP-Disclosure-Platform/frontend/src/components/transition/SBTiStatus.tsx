/**
 * SBTiStatus - SBTi alignment and target status display
 *
 * Shows whether the organization's transition plan is aligned with
 * SBTi requirements (1.5C pathway), current SBTi status, board
 * oversight, and public disclosure status.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Divider } from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Science,
  Visibility,
  SupervisedUserCircle,
  TrendingDown,
} from '@mui/icons-material';

interface SBTiStatusProps {
  sbtiAligned: boolean;
  sbtiStatus: string;
  pathwayType: string;
  reductionTargetPct: number;
  annualReductionRate: number;
  boardOversight: boolean;
  publiclyDisclosed: boolean;
  targetYear: number;
}

const SBTiStatus: React.FC<SBTiStatusProps> = ({
  sbtiAligned,
  sbtiStatus,
  pathwayType,
  reductionTargetPct,
  annualReductionRate,
  boardOversight,
  publiclyDisclosed,
  targetYear,
}) => {
  const checks = [
    {
      label: 'SBTi Aligned (1.5C)',
      met: sbtiAligned,
      detail: `Status: ${sbtiStatus}`,
      icon: <Science sx={{ fontSize: 20 }} />,
    },
    {
      label: 'Board Oversight',
      met: boardOversight,
      detail: boardOversight
        ? 'Board has oversight of transition plan'
        : 'No board oversight reported',
      icon: <SupervisedUserCircle sx={{ fontSize: 20 }} />,
    },
    {
      label: 'Publicly Disclosed',
      met: publiclyDisclosed,
      detail: publiclyDisclosed
        ? 'Transition plan is publicly available'
        : 'Plan not publicly disclosed',
      icon: <Visibility sx={{ fontSize: 20 }} />,
    },
    {
      label: 'Sufficient Reduction Rate',
      met: annualReductionRate >= 4.2,
      detail: `${annualReductionRate.toFixed(1)}% per year (>= 4.2% required for 1.5C)`,
      icon: <TrendingDown sx={{ fontSize: 20 }} />,
    },
  ];

  const metCount = checks.filter((c) => c.met).length;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">SBTi Alignment</Typography>
          <Chip
            label={`${metCount}/${checks.length} met`}
            size="small"
            color={metCount === checks.length ? 'success' : metCount >= 2 ? 'warning' : 'error'}
          />
        </Box>

        {/* Summary metrics */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
          <Chip label={`Pathway: ${pathwayType}`} size="small" variant="outlined" />
          <Chip
            label={`Target: -${reductionTargetPct.toFixed(0)}% by ${targetYear}`}
            size="small"
            variant="outlined"
          />
          <Chip
            label={`SBTi Status: ${sbtiStatus}`}
            size="small"
            color={sbtiAligned ? 'success' : 'default'}
          />
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* Requirement checklist */}
        {checks.map((check, idx) => (
          <Box
            key={check.label}
            sx={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 1.5,
              py: 1,
              borderBottom: idx < checks.length - 1 ? '1px solid #f5f5f5' : 'none',
            }}
          >
            {check.met ? (
              <CheckCircle sx={{ color: '#2e7d32', fontSize: 22, mt: 0.25 }} />
            ) : (
              <Cancel sx={{ color: '#e53935', fontSize: 22, mt: 0.25 }} />
            )}
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {check.icon}
                <Typography variant="body2" fontWeight={600}>
                  {check.label}
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {check.detail}
              </Typography>
            </Box>
          </Box>
        ))}
      </CardContent>
    </Card>
  );
};

export default SBTiStatus;
