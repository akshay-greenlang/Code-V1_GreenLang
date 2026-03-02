/**
 * VerificationTimeline - Verification workflow stepper
 *
 * MUI Stepper showing verification stages: Draft, Internal Review,
 * Approved, External Verification, Verified. Shows current stage,
 * finding counts per stage, and action buttons for advancing.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Chip,
  Grid,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Divider,
} from '@mui/material';
import {
  CheckCircle,
  Edit,
  RateReview,
  Verified,
  AssignmentTurnedIn,
  PersonAdd,
} from '@mui/icons-material';
import type { VerificationRecord, VerificationStage, Finding } from '../../types';
import { formatDate } from '../../utils/formatters';
import DataTable, { Column } from '../common/DataTable';

interface VerificationTimelineProps {
  verification: VerificationRecord | null;
  onAdvance: (action: string) => void;
  onAssignVerifier: (verifier: string) => void;
}

const STAGE_ORDER: VerificationStage[] = [
  'draft' as VerificationStage,
  'internal_review' as VerificationStage,
  'approved' as VerificationStage,
  'external_verification' as VerificationStage,
  'verified' as VerificationStage,
];

const STAGE_LABELS: Record<string, string> = {
  draft: 'Draft',
  internal_review: 'Internal Review',
  approved: 'Approved',
  external_verification: 'External Verification',
  verified: 'Verified',
};

const STAGE_ICONS: Record<string, React.ReactNode> = {
  draft: <Edit />,
  internal_review: <RateReview />,
  approved: <AssignmentTurnedIn />,
  external_verification: <Verified />,
  verified: <CheckCircle />,
};

const STAGE_ACTIONS: Record<string, string> = {
  draft: 'Start Internal Review',
  internal_review: 'Approve',
  approved: 'Assign Verifier',
  external_verification: 'Complete Verification',
};

const VerificationTimeline: React.FC<VerificationTimelineProps> = ({
  verification,
  onAdvance,
  onAssignVerifier,
}) => {
  const [assignDialogOpen, setAssignDialogOpen] = useState(false);
  const [verifierName, setVerifierName] = useState('');

  if (!verification) {
    return (
      <Alert severity="info">
        No verification engagement started. Click "Start Verification" to begin.
      </Alert>
    );
  }

  const currentStageIndex = STAGE_ORDER.indexOf(verification.status as VerificationStage);
  const findingSummary = verification.findings_summary;

  const handleAction = () => {
    if (verification.status === 'approved') {
      setAssignDialogOpen(true);
    } else {
      const nextActions: Record<string, string> = {
        draft: 'start_review',
        internal_review: 'approve',
        not_started: 'start_review',
        in_progress: 'approve',
        external_verification: 'complete',
      };
      onAdvance(nextActions[verification.status] || 'advance');
    }
  };

  const handleAssignVerifier = () => {
    onAssignVerifier(verifierName);
    setAssignDialogOpen(false);
    setVerifierName('');
  };

  // Finding table columns
  const findingColumns: Column<Finding>[] = [
    {
      id: 'severity',
      label: 'Severity',
      render: (row) => (
        <Chip
          label={row.severity}
          size="small"
          color={
            row.severity === 'critical' ? 'error' :
            row.severity === 'high' ? 'warning' :
            row.severity === 'medium' ? 'info' : 'default'
          }
        />
      ),
    },
    { id: 'category', label: 'Category' },
    {
      id: 'description',
      label: 'Description',
      render: (row) => (
        <Typography variant="body2" sx={{ maxWidth: 300 }} noWrap>
          {row.description}
        </Typography>
      ),
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => (
        <Chip
          label={row.status}
          size="small"
          color={row.status === 'resolved' || row.status === 'accepted' ? 'success' : 'warning'}
          variant="outlined"
        />
      ),
    },
    {
      id: 'created_at',
      label: 'Date',
      render: (row) => formatDate(row.created_at),
    },
  ];

  return (
    <Box>
      {/* Verification header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box>
              <Typography variant="h6">Verification Workflow</Typography>
              {verification.verifier_name && (
                <Typography variant="body2" color="text.secondary">
                  Verifier: {verification.verifier_name} ({verification.verifier_accreditation})
                </Typography>
              )}
            </Box>
            <Chip
              label={STAGE_LABELS[verification.status] || verification.status}
              color="primary"
              variant="filled"
            />
          </Box>

          {/* Stepper */}
          <Stepper activeStep={currentStageIndex} alternativeLabel sx={{ mb: 3 }}>
            {STAGE_ORDER.map((stage, index) => (
              <Step key={stage} completed={index < currentStageIndex}>
                <StepLabel
                  icon={STAGE_ICONS[stage]}
                  optional={
                    <Typography variant="caption" color="text.secondary">
                      {index < currentStageIndex ? 'Complete' : index === currentStageIndex ? 'Current' : 'Pending'}
                    </Typography>
                  }
                >
                  {STAGE_LABELS[stage]}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          {/* Action button */}
          {currentStageIndex < STAGE_ORDER.length - 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <Button variant="contained" onClick={handleAction}>
                {STAGE_ACTIONS[STAGE_ORDER[currentStageIndex]] || 'Advance'}
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Findings summary */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Findings Summary</Typography>
          <Grid container spacing={2}>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="text.secondary">Total</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{findingSummary.total_findings}</Typography>
            </Grid>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="error.main">Critical</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'error.main' }}>{findingSummary.critical_count}</Typography>
            </Grid>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="warning.main">High</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'warning.main' }}>{findingSummary.high_count}</Typography>
            </Grid>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="info.main">Medium</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'info.main' }}>{findingSummary.medium_count}</Typography>
            </Grid>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="text.secondary">Open</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{findingSummary.open_count}</Typography>
            </Grid>
            <Grid item xs={4} sm={2}>
              <Typography variant="caption" color="success.main">Resolved</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'success.main' }}>{findingSummary.resolved_count}</Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Findings table */}
      {verification.findings.length > 0 && (
        <DataTable
          columns={findingColumns}
          rows={verification.findings}
          rowKey={(row) => row.id}
          searchPlaceholder="Search findings..."
          defaultSort="severity"
          defaultOrder="desc"
        />
      )}

      {/* Assign verifier dialog */}
      <Dialog open={assignDialogOpen} onClose={() => setAssignDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Assign External Verifier</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Verifier Name / Organization"
            value={verifierName}
            onChange={(e) => setVerifierName(e.target.value)}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAssignDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleAssignVerifier} disabled={!verifierName.trim()}>
            Assign
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VerificationTimeline;
