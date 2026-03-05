/**
 * Verification Page - Verification management
 *
 * Composes VerificationStatus, CoverageTracker, VerifierDetails,
 * and AssuranceLevel for managing third-party verification of
 * emissions data.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Button,
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
import { Add, VerifiedUser } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchVerificationRecords,
  createVerification,
  fetchVerificationSummary,
  deleteVerification,
} from '../store/slices/verificationSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import VerificationStatus from '../components/verification/VerificationStatus';
import CoverageTracker from '../components/verification/CoverageTracker';
import VerifierDetails from '../components/verification/VerifierDetails';
import AssuranceLevel from '../components/verification/AssuranceLevel';
import { VerificationLevel } from '../types';

const DEMO_ORG_ID = 'demo-org';

const VerificationPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { records, summary, loading, error } = useAppSelector(
    (s) => s.verification,
  );

  const [addOpen, setAddOpen] = useState(false);
  const [newRecord, setNewRecord] = useState({
    scope: 'Scope 1',
    verifier_name: '',
    verifier_accreditation: '',
    verification_level: VerificationLevel.LIMITED as VerificationLevel,
  });

  useEffect(() => {
    dispatch(fetchVerificationRecords(DEMO_ORG_ID));
    dispatch(fetchVerificationSummary(DEMO_ORG_ID));
  }, [dispatch]);

  const handleAdd = () => {
    dispatch(createVerification({
      orgId: DEMO_ORG_ID,
      data: newRecord,
    }));
    setAddOpen(false);
    setNewRecord({
      scope: 'Scope 1',
      verifier_name: '',
      verifier_accreditation: '',
      verification_level: VerificationLevel.LIMITED,
    });
  };

  const handleDelete = (id: string) => {
    dispatch(deleteVerification({ orgId: DEMO_ORG_ID, recordId: id }));
  };

  if (loading && records.length === 0) {
    return <LoadingSpinner message="Loading verification data..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Verification Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage third-party verification of emissions data for CDP scoring
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setAddOpen(true)}
        >
          Add Verification
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Verification status overview */}
        {summary && (
          <Grid item xs={12}>
            <VerificationStatus summary={summary} />
          </Grid>
        )}

        {/* Coverage tracker */}
        <Grid item xs={12} md={6}>
          <CoverageTracker records={records} />
        </Grid>

        {/* Assurance levels */}
        <Grid item xs={12} md={6}>
          <AssuranceLevel records={records} />
        </Grid>

        {/* Individual verifier cards */}
        {records.map((record) => (
          <Grid item xs={12} md={6} key={record.id}>
            <VerifierDetails record={record} onDelete={handleDelete} />
          </Grid>
        ))}

        {records.length === 0 && (
          <Grid item xs={12}>
            <Alert severity="info">
              No verification records found. Add a verification record to track
              third-party assurance of your emissions data.
            </Alert>
          </Grid>
        )}
      </Grid>

      {/* Add verification dialog */}
      <Dialog open={addOpen} onClose={() => setAddOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <VerifiedUser />
            Add Verification Record
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Scope</InputLabel>
              <Select
                value={newRecord.scope}
                label="Scope"
                onChange={(e) => setNewRecord({ ...newRecord, scope: e.target.value })}
              >
                <MenuItem value="Scope 1">Scope 1</MenuItem>
                <MenuItem value="Scope 2">Scope 2</MenuItem>
                <MenuItem value="Scope 3">Scope 3</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Verifier Name"
              value={newRecord.verifier_name}
              onChange={(e) => setNewRecord({ ...newRecord, verifier_name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Accreditation"
              value={newRecord.verifier_accreditation}
              onChange={(e) => setNewRecord({ ...newRecord, verifier_accreditation: e.target.value })}
              fullWidth
              placeholder="e.g., ISO 14065, ISAE 3410"
            />
            <FormControl fullWidth>
              <InputLabel>Assurance Level</InputLabel>
              <Select
                value={newRecord.verification_level}
                label="Assurance Level"
                onChange={(e) => setNewRecord({
                  ...newRecord,
                  verification_level: e.target.value as VerificationLevel,
                })}
              >
                <MenuItem value={VerificationLevel.LIMITED}>Limited Assurance</MenuItem>
                <MenuItem value={VerificationLevel.REASONABLE}>Reasonable Assurance</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAdd}
            disabled={!newRecord.verifier_name}
          >
            Add Record
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VerificationPage;
