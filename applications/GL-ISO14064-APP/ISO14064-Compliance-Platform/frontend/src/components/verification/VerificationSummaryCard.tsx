/**
 * VerificationSummaryCard - Verification overview card
 *
 * Displays verifier name, accreditation, verification level,
 * opinion, and finding count summary.
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Chip,
  Divider,
} from '@mui/material';
import {
  Person,
  VerifiedUser,
  Assessment,
  Warning,
  Error as ErrorIcon,
  Info,
  BugReport,
} from '@mui/icons-material';
import type { VerificationRecord } from '../../types';
import { formatDate } from '../../utils/formatters';
import StatusChip from '../common/StatusChip';

interface VerificationSummaryCardProps {
  verification: VerificationRecord;
}

const VerificationSummaryCard: React.FC<VerificationSummaryCardProps> = ({
  verification,
}) => {
  const fs = verification.findings_summary;

  return (
    <Card>
      <CardHeader
        title="Verification Overview"
        action={<StatusChip status={verification.stage} />}
      />
      <CardContent>
        <Grid container spacing={2}>
          {/* Verifier Info */}
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Person fontSize="small" color="action" />
              <Typography variant="body2" color="text.secondary">
                Verifier
              </Typography>
            </Box>
            <Typography variant="body1" fontWeight={600}>
              {verification.verifier_name || 'Not assigned'}
            </Typography>
            {verification.verifier_accreditation && (
              <Typography variant="caption" color="text.secondary">
                {verification.verifier_accreditation}
              </Typography>
            )}
          </Grid>

          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <VerifiedUser fontSize="small" color="action" />
              <Typography variant="body2" color="text.secondary">
                Verification Level
              </Typography>
            </Box>
            <Typography variant="body1" fontWeight={600}>
              {verification.verification_level.replace(/_/g, ' ')}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Scope: {verification.scope_of_verification || 'Full inventory'}
            </Typography>
          </Grid>

          {/* Opinion */}
          {verification.opinion && (
            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Assessment fontSize="small" color="action" />
                <Typography variant="body2" color="text.secondary">
                  Verification Opinion
                </Typography>
              </Box>
              <Typography variant="body1" sx={{ fontStyle: 'italic' }}>
                "{verification.opinion}"
              </Typography>
              {verification.opinion_date && (
                <Typography variant="caption" color="text.secondary">
                  Issued: {formatDate(verification.opinion_date)}
                </Typography>
              )}
            </Grid>
          )}

          {/* Findings Summary */}
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" gutterBottom>
              Findings Summary
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                icon={<BugReport />}
                label={`Total: ${fs.total_findings}`}
                size="small"
                variant="outlined"
              />
              {fs.critical_count > 0 && (
                <Chip
                  icon={<ErrorIcon />}
                  label={`Critical: ${fs.critical_count}`}
                  size="small"
                  color="error"
                />
              )}
              {fs.high_count > 0 && (
                <Chip
                  icon={<Warning />}
                  label={`High: ${fs.high_count}`}
                  size="small"
                  color="error"
                  variant="outlined"
                />
              )}
              {fs.medium_count > 0 && (
                <Chip
                  icon={<Warning />}
                  label={`Medium: ${fs.medium_count}`}
                  size="small"
                  color="warning"
                />
              )}
              {fs.low_count > 0 && (
                <Chip
                  icon={<Info />}
                  label={`Low: ${fs.low_count}`}
                  size="small"
                  color="info"
                />
              )}
              <Chip
                label={`Open: ${fs.open_count}`}
                size="small"
                color={fs.open_count > 0 ? 'error' : 'success'}
                variant="outlined"
              />
              <Chip
                label={`Resolved: ${fs.resolved_count}`}
                size="small"
                color="success"
                variant="outlined"
              />
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default VerificationSummaryCard;
