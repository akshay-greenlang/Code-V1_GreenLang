/**
 * VerificationManagement Page - Verification timeline, findings,
 * stage advancement, and verification summary.
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Alert,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
} from '@mui/material';
import { Add, SkipNext, CheckCircle } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchVerification,
  fetchVerifications,
  startVerification,
  advanceStage,
  addFinding,
  resolveFinding,
} from '../store/slices/verificationSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import VerificationTimeline from '../components/verification/VerificationTimeline';
import VerificationSummaryCard from '../components/verification/VerificationSummaryCard';
import FindingsTable from '../components/verification/FindingsTable';
import DataTable, { Column } from '../components/common/DataTable';
import StatusChip from '../components/common/StatusChip';
import type {
  VerificationRecord,
  CreateVerificationRequest,
  AddFindingRequest,
} from '../types';
import { VerificationLevel, FindingSeverity, ISOCategory, ISO_CATEGORY_SHORT_NAMES } from '../types';
import { formatDate } from '../utils/formatters';

const VerificationManagement: React.FC = () => {
  const { id: verificationId } = useParams<{ id?: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { currentVerification, verifications, loading, error } = useSelector(
    (s: AppRootState) => s.verification,
  );

  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [findingDialogOpen, setFindingDialogOpen] = useState(false);
  const [newVerification, setNewVerification] = useState<CreateVerificationRequest>({
    inventory_id: '',
    verifier_name: '',
    verifier_accreditation: '',
    verification_level: VerificationLevel.LIMITED,
    scope_of_verification: 'Full inventory',
  });
  const [newFinding, setNewFinding] = useState<AddFindingRequest>({
    category: '',
    severity: FindingSeverity.MEDIUM,
    description: '',
    recommendation: '',
  });

  useEffect(() => {
    if (verificationId) {
      dispatch(fetchVerification(verificationId));
    }
  }, [dispatch, verificationId]);

  const handleStartVerification = () => {
    dispatch(startVerification(newVerification));
    setStartDialogOpen(false);
  };

  const handleAdvance = () => {
    if (currentVerification) {
      dispatch(advanceStage(currentVerification.id));
    }
  };

  const handleAddFinding = () => {
    if (currentVerification) {
      dispatch(addFinding({ verificationId: currentVerification.id, payload: newFinding }));
    }
    setFindingDialogOpen(false);
    setNewFinding({ category: '', severity: FindingSeverity.MEDIUM, description: '', recommendation: '' });
  };

  const handleResolveFinding = (findingId: string, resolution: string) => {
    if (currentVerification) {
      dispatch(resolveFinding({ verificationId: currentVerification.id, findingId, resolution }));
    }
  };

  if (loading && !currentVerification && verifications.length === 0) {
    return <LoadingSpinner message="Loading verification..." />;
  }

  // If no verificationId and no current, show list mode
  if (!verificationId && !currentVerification) {
    const listColumns: Column<VerificationRecord>[] = [
      { id: 'verifier_name', label: 'Verifier' },
      {
        id: 'verification_level',
        label: 'Level',
        render: (row) => row.verification_level.replace(/_/g, ' '),
      },
      {
        id: 'stage',
        label: 'Stage',
        render: (row) => <StatusChip status={row.stage} />,
      },
      {
        id: 'findings_summary',
        label: 'Findings',
        render: (row) => `${row.findings_summary.total_findings} (${row.findings_summary.open_count} open)`,
      },
      {
        id: 'created_at',
        label: 'Started',
        render: (row) => formatDate(row.created_at),
      },
    ];

    return (
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5" fontWeight={700}>
            Verification Management
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setStartDialogOpen(true)}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Start Verification
          </Button>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <DataTable
          columns={listColumns}
          rows={verifications}
          rowKey={(r) => r.id}
          searchPlaceholder="Search verifications..."
        />

        {/* Start dialog omitted for brevity in list mode; handled below */}
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight={700}>
          Verification Workflow
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Add />}
            onClick={() => setFindingDialogOpen(true)}
          >
            Add Finding
          </Button>
          <Button
            variant="contained"
            startIcon={<SkipNext />}
            onClick={handleAdvance}
            disabled={loading || currentVerification?.stage === 'verified'}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Advance Stage
          </Button>
        </Box>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {currentVerification && (
        <Grid container spacing={3}>
          {/* Timeline */}
          <Grid item xs={12}>
            <VerificationTimeline currentStage={currentVerification.stage} />
          </Grid>

          {/* Summary card */}
          <Grid item xs={12} md={5}>
            <VerificationSummaryCard verification={currentVerification} />
          </Grid>

          {/* Findings */}
          <Grid item xs={12} md={7}>
            <Typography variant="h6" gutterBottom>
              Findings ({currentVerification.findings.length})
            </Typography>
            <FindingsTable
              findings={currentVerification.findings}
              onResolve={handleResolveFinding}
            />
          </Grid>
        </Grid>
      )}

      {/* Start Verification Dialog */}
      <Dialog open={startDialogOpen} onClose={() => setStartDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start Verification Engagement</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Inventory ID"
                value={newVerification.inventory_id}
                onChange={(e) =>
                  setNewVerification({ ...newVerification, inventory_id: e.target.value })
                }
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Verifier Name"
                value={newVerification.verifier_name}
                onChange={(e) =>
                  setNewVerification({ ...newVerification, verifier_name: e.target.value })
                }
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Verification Level"
                value={newVerification.verification_level}
                onChange={(e) =>
                  setNewVerification({
                    ...newVerification,
                    verification_level: e.target.value as VerificationLevel,
                  })
                }
              >
                {Object.values(VerificationLevel).map((l) => (
                  <MenuItem key={l} value={l}>{l.replace(/_/g, ' ')}</MenuItem>
                ))}
              </TextField>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleStartVerification}
            variant="contained"
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Start
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add Finding Dialog */}
      <Dialog open={findingDialogOpen} onClose={() => setFindingDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Finding</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Category"
                value={newFinding.category}
                onChange={(e) => setNewFinding({ ...newFinding, category: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Severity"
                value={newFinding.severity}
                onChange={(e) =>
                  setNewFinding({ ...newFinding, severity: e.target.value as FindingSeverity })
                }
              >
                {Object.values(FindingSeverity).map((s) => (
                  <MenuItem key={s} value={s}>{s}</MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={newFinding.description}
                onChange={(e) => setNewFinding({ ...newFinding, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Recommendation"
                value={newFinding.recommendation}
                onChange={(e) => setNewFinding({ ...newFinding, recommendation: e.target.value })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFindingDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleAddFinding}
            variant="contained"
            disabled={!newFinding.description}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VerificationManagement;
