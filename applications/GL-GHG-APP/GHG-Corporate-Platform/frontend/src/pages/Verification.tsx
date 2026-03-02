/**
 * Verification Page - Third-party verification workflow
 *
 * Composes VerificationTimeline stepper, finding tracker table,
 * verification history, and start review / assign verifier buttons.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Alert,
  Grid,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { PlayArrow, VerifiedUser } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import {
  fetchVerifications,
  fetchVerification,
  startVerification,
  approveVerification,
  addFinding,
} from '../store/slices/verificationSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import VerificationTimeline from '../components/verification/VerificationTimeline';
import StatusBadge from '../components/common/StatusBadge';
import DataTable, { Column } from '../components/common/DataTable';
import { formatDate } from '../utils/formatters';
import type { VerificationRecord, StartVerificationRequest, VerificationLevel, Scope } from '../types';

const DEMO_INVENTORY_ID = 'demo-inventory';

const VerificationPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { currentVerification, verifications, loading, error } = useAppSelector(
    (state) => state.verification
  );
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [startForm, setStartForm] = useState({
    verifier_name: '',
    verifier_accreditation: '',
    verification_level: 'limited' as VerificationLevel,
  });

  useEffect(() => {
    dispatch(fetchVerifications(DEMO_INVENTORY_ID));
  }, [dispatch]);

  const handleStartVerification = () => {
    const payload: StartVerificationRequest = {
      inventory_id: DEMO_INVENTORY_ID,
      verifier_name: startForm.verifier_name,
      verifier_accreditation: startForm.verifier_accreditation,
      verification_level: startForm.verification_level,
      scope_covered: ['scope_1', 'scope_2'] as Scope[],
    };
    dispatch(startVerification(payload));
    setStartDialogOpen(false);
  };

  const handleAdvance = (action: string) => {
    if (currentVerification) {
      dispatch(approveVerification({ verificationId: currentVerification.id, opinion: action }));
    }
  };

  const handleAssignVerifier = (verifier: string) => {
    // In production, dispatch assignVerifier
  };

  if (loading && verifications.length === 0) return <LoadingSpinner message="Loading verification..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  // History table columns
  const historyColumns: Column<VerificationRecord>[] = [
    {
      id: 'verifier_name',
      label: 'Verifier',
      render: (row) => row.verifier_name,
    },
    {
      id: 'verification_level',
      label: 'Level',
      render: (row) => <StatusBadge status={row.verification_level} />,
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
    },
    {
      id: 'findings',
      label: 'Findings',
      align: 'center',
      render: (row) => String(row.findings_summary.total_findings),
      getValue: (row) => row.findings_summary.total_findings,
    },
    {
      id: 'start_date',
      label: 'Start Date',
      render: (row) => formatDate(row.start_date),
    },
    {
      id: 'end_date',
      label: 'End Date',
      render: (row) => row.end_date ? formatDate(row.end_date) : 'In Progress',
    },
  ];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Verification Workflow</Typography>
        <Button
          variant="contained"
          startIcon={<PlayArrow />}
          onClick={() => setStartDialogOpen(true)}
        >
          Start Verification
        </Button>
      </Box>

      {/* Active verification timeline */}
      <Box sx={{ mb: 3 }}>
        <VerificationTimeline
          verification={currentVerification}
          onAdvance={handleAdvance}
          onAssignVerifier={handleAssignVerifier}
        />
      </Box>

      {/* Verification history */}
      {verifications.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Verification History
          </Typography>
          <DataTable
            columns={historyColumns}
            rows={verifications}
            rowKey={(row) => row.id}
            searchPlaceholder="Search verifications..."
            onRowClick={(row) => dispatch(fetchVerification(row.id))}
          />
        </Box>
      )}

      {/* Start verification dialog */}
      <Dialog open={startDialogOpen} onClose={() => setStartDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start Verification Engagement</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '16px !important' }}>
          <TextField
            label="Verifier Name / Organization"
            fullWidth
            value={startForm.verifier_name}
            onChange={(e) => setStartForm({ ...startForm, verifier_name: e.target.value })}
          />
          <TextField
            label="Accreditation"
            fullWidth
            value={startForm.verifier_accreditation}
            onChange={(e) => setStartForm({ ...startForm, verifier_accreditation: e.target.value })}
          />
          <FormControl fullWidth>
            <InputLabel>Verification Level</InputLabel>
            <Select
              value={startForm.verification_level}
              label="Verification Level"
              onChange={(e) => setStartForm({ ...startForm, verification_level: e.target.value as VerificationLevel })}
            >
              <MenuItem value="limited">Limited Assurance</MenuItem>
              <MenuItem value="reasonable">Reasonable Assurance</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleStartVerification}
            disabled={!startForm.verifier_name}
          >
            Start
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VerificationPage;
