/**
 * DDSValidation - Validation results display for a Due Diligence Statement.
 *
 * Shows overall pass/fail status, completeness percentage, per-section
 * validation checks, missing data highlights, and recommendations.
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Divider,
  Button,
  Alert,
  Paper,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import BuildIcon from '@mui/icons-material/Build';
import type { DDSValidationResult, DDSValidationIssue } from '../../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DDSValidationProps {
  validation: DDSValidationResult;
  onFixIssues?: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function severityIcon(severity: DDSValidationIssue['severity']) {
  switch (severity) {
    case 'error':
      return <CancelIcon color="error" fontSize="small" />;
    case 'warning':
      return <WarningAmberIcon color="warning" fontSize="small" />;
    case 'info':
      return <CheckCircleIcon color="info" fontSize="small" />;
  }
}

function severityColor(severity: DDSValidationIssue['severity']): 'error' | 'warning' | 'info' {
  return severity;
}

/** Group issues by field/section. */
function groupIssues(issues: DDSValidationIssue[]): Record<string, DDSValidationIssue[]> {
  const groups: Record<string, DDSValidationIssue[]> = {};
  for (const issue of issues) {
    const section = issue.field || 'general';
    if (!groups[section]) groups[section] = [];
    groups[section].push(issue);
  }
  return groups;
}

const SECTION_LABELS: Record<string, string> = {
  operator_info: 'Operator Information',
  product_description: 'Product Description',
  country_of_production: 'Country of Production',
  geolocation: 'Geolocation Data',
  risk_assessment: 'Risk Assessment',
  risk_mitigation: 'Risk Mitigation',
  conclusion: 'Conclusion',
  general: 'General',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DDSValidation: React.FC<DDSValidationProps> = ({
  validation,
  onFixIssues,
}) => {
  const errorCount = validation.issues.filter((i) => i.severity === 'error').length;
  const warningCount = validation.issues.filter((i) => i.severity === 'warning').length;
  const infoCount = validation.issues.filter((i) => i.severity === 'info').length;
  const groupedIssues = groupIssues(validation.issues);

  // Field validation summary
  const fieldEntries = Object.entries(validation.field_validations);
  const passedFields = fieldEntries.filter(([, v]) => v).length;
  const totalFields = fieldEntries.length;

  return (
    <Card>
      <CardContent>
        {/* Overall Status */}
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
          <Stack direction="row" alignItems="center" spacing={1.5}>
            {validation.is_valid ? (
              <CheckCircleIcon sx={{ fontSize: 36 }} color="success" />
            ) : (
              <CancelIcon sx={{ fontSize: 36 }} color="error" />
            )}
            <Box>
              <Typography variant="h6">
                {validation.is_valid ? 'Validation Passed' : 'Validation Failed'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                DDS {validation.dds_id}
              </Typography>
            </Box>
          </Stack>
          <Paper variant="outlined" sx={{ px: 2, py: 1, textAlign: 'center' }}>
            <Typography variant="h4" color={validation.completeness_score >= 90 ? 'success.main' : 'warning.main'}>
              {validation.completeness_score.toFixed(0)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">Completeness</Typography>
          </Paper>
        </Stack>

        {/* Progress bar */}
        <LinearProgress
          variant="determinate"
          value={validation.completeness_score}
          color={validation.completeness_score >= 90 ? 'success' : validation.completeness_score >= 70 ? 'warning' : 'error'}
          sx={{ height: 8, borderRadius: 1, mb: 2 }}
        />

        {/* Issue counts */}
        <Stack direction="row" spacing={1} mb={2}>
          {errorCount > 0 && (
            <Chip
              icon={<CancelIcon />}
              label={`${errorCount} Error${errorCount > 1 ? 's' : ''}`}
              color="error"
              variant="outlined"
              size="small"
            />
          )}
          {warningCount > 0 && (
            <Chip
              icon={<WarningAmberIcon />}
              label={`${warningCount} Warning${warningCount > 1 ? 's' : ''}`}
              color="warning"
              variant="outlined"
              size="small"
            />
          )}
          {infoCount > 0 && (
            <Chip
              icon={<CheckCircleIcon />}
              label={`${infoCount} Info`}
              color="info"
              variant="outlined"
              size="small"
            />
          )}
          <Chip
            label={`${passedFields}/${totalFields} fields passed`}
            variant="outlined"
            size="small"
          />
        </Stack>

        <Divider sx={{ mb: 2 }} />

        {/* Per-section checks */}
        <Typography variant="subtitle2" gutterBottom>
          Section Checks
        </Typography>

        {/* Sections with no issues */}
        {fieldEntries
          .filter(([field, passed]) => passed && !groupedIssues[field])
          .map(([field]) => (
            <Stack key={field} direction="row" alignItems="center" spacing={1} sx={{ py: 0.5 }}>
              <CheckCircleIcon color="success" fontSize="small" />
              <Typography variant="body2">
                {SECTION_LABELS[field] ?? field.replace('_', ' ')}
              </Typography>
            </Stack>
          ))}

        {/* Sections with issues */}
        {Object.entries(groupedIssues).map(([section, issues]) => (
          <Box key={section} sx={{ mt: 1 }}>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
              {issues.some((i) => i.severity === 'error') ? (
                <CancelIcon color="error" fontSize="small" />
              ) : issues.some((i) => i.severity === 'warning') ? (
                <WarningAmberIcon color="warning" fontSize="small" />
              ) : (
                <CheckCircleIcon color="info" fontSize="small" />
              )}
              <Typography variant="body2" fontWeight={600}>
                {SECTION_LABELS[section] ?? section.replace('_', ' ')}
              </Typography>
            </Stack>

            <List dense disablePadding sx={{ pl: 4 }}>
              {issues.map((issue, idx) => (
                <ListItem key={idx} disableGutters sx={{ py: 0.25 }}>
                  <ListItemIcon sx={{ minWidth: 28 }}>
                    {severityIcon(issue.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={issue.message}
                    secondary={issue.code}
                    primaryTypographyProps={{
                      variant: 'body2',
                      color: issue.severity === 'error' ? 'error.main' : 'text.primary',
                    }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        ))}

        {/* Recommendation */}
        {validation.recommendation && (
          <>
            <Divider sx={{ my: 2 }} />
            <Alert severity={validation.is_valid ? 'success' : 'warning'} variant="outlined">
              <Typography variant="body2">{validation.recommendation}</Typography>
            </Alert>
          </>
        )}

        {/* Fix Issues button */}
        {!validation.is_valid && onFixIssues && (
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<BuildIcon />}
              onClick={onFixIssues}
            >
              Fix Issues
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default DDSValidation;
