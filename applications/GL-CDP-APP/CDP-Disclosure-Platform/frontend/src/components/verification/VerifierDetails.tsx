/**
 * VerifierDetails - Verification record detail card
 *
 * Shows detailed information for a single verification record
 * including verifier name, accreditation, scope, coverage,
 * level, and linked verification statement.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  IconButton,
  Divider,
} from '@mui/material';
import {
  Business,
  CheckCircle,
  Cancel,
  OpenInNew,
  Delete,
  CalendarToday,
} from '@mui/icons-material';
import type { VerificationRecord } from '../../types';
import { VerificationLevel } from '../../types';
import { formatDate } from '../../utils/formatters';

interface VerifierDetailsProps {
  record: VerificationRecord;
  onDelete?: (id: string) => void;
}

const VerifierDetails: React.FC<VerifierDetailsProps> = ({ record, onDelete }) => {
  const levelColor = record.verification_level === VerificationLevel.REASONABLE
    ? '#1b5e20'
    : record.verification_level === VerificationLevel.LIMITED
      ? '#ef6c00'
      : '#9e9e9e';

  return (
    <Card sx={{ border: `1px solid ${record.verified ? '#c8e6c9' : '#e0e0e0'}` }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Business sx={{ color: '#1565c0', fontSize: 22 }} />
            <Typography variant="subtitle1" fontWeight={600}>
              {record.verifier_name}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {record.verified ? (
              <CheckCircle sx={{ color: '#2e7d32', fontSize: 20 }} />
            ) : (
              <Cancel sx={{ color: '#9e9e9e', fontSize: 20 }} />
            )}
            {onDelete && (
              <IconButton
                size="small"
                onClick={() => onDelete(record.id)}
                sx={{ color: '#9e9e9e', '&:hover': { color: '#e53935' } }}
              >
                <Delete fontSize="small" />
              </IconButton>
            )}
          </Box>
        </Box>

        {record.verifier_accreditation && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Accreditation: {record.verifier_accreditation}
          </Typography>
        )}

        <Divider sx={{ my: 1.5 }} />

        <Box sx={{ display: 'flex', gap: 3, mb: 1.5 }}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Scope
            </Typography>
            <Typography variant="body2" fontWeight={500}>
              {record.scope}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Coverage
            </Typography>
            <Typography variant="body2" fontWeight={500}>
              {record.coverage_pct.toFixed(0)}%
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Level
            </Typography>
            <Chip
              label={record.verification_level.replace('_', ' ')}
              size="small"
              sx={{
                fontSize: 10,
                bgcolor: levelColor + '15',
                color: levelColor,
                border: `1px solid ${levelColor}40`,
              }}
            />
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Year
            </Typography>
            <Typography variant="body2" fontWeight={500}>
              {record.reporting_year}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <CalendarToday sx={{ fontSize: 14, color: '#9e9e9e' }} />
            <Typography variant="caption" color="text.secondary">
              {record.verification_date
                ? `Verified: ${formatDate(record.verification_date)}`
                : 'Date pending'}
            </Typography>
          </Box>

          {record.statement_url && (
            <Button
              size="small"
              variant="outlined"
              href={record.statement_url}
              target="_blank"
              endIcon={<OpenInNew sx={{ fontSize: 14 }} />}
              sx={{ fontSize: 11 }}
            >
              Verification Statement
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default VerifierDetails;
