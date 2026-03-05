/**
 * SupplyChain Page - Supply chain engagement management
 *
 * Composes SupplierList, EngagementDashboard, HotspotMap,
 * and SupplierTracker for end-to-end supply chain management.
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
} from '@mui/material';
import { PersonAdd } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchSuppliers,
  fetchSupplyChainSummary,
  inviteSupplier,
} from '../store/slices/supplyChainSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import SupplierList from '../components/supply_chain/SupplierList';
import EngagementDashboard from '../components/supply_chain/EngagementDashboard';
import HotspotMap from '../components/supply_chain/HotspotMap';
import SupplierTracker from '../components/supply_chain/SupplierTracker';
import type { SupplierRequest } from '../types';

const DEMO_ORG_ID = 'demo-org';

const SupplyChainPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { suppliers, summary, loading, error } = useAppSelector(
    (s) => s.supplyChain,
  );

  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteForm, setInviteForm] = useState({
    supplier_name: '',
    supplier_email: '',
    supplier_country: '',
    supplier_sector: '',
    message: '',
  });
  const [selectedSupplier, setSelectedSupplier] = useState<SupplierRequest | null>(null);

  useEffect(() => {
    dispatch(fetchSuppliers(DEMO_ORG_ID));
    dispatch(fetchSupplyChainSummary(DEMO_ORG_ID));
  }, [dispatch]);

  const handleInvite = () => {
    dispatch(inviteSupplier({
      orgId: DEMO_ORG_ID,
      data: inviteForm,
    }));
    setInviteOpen(false);
    setInviteForm({
      supplier_name: '',
      supplier_email: '',
      supplier_country: '',
      supplier_sector: '',
      message: '',
    });
  };

  const handleSupplierSelect = (supplier: SupplierRequest) => {
    setSelectedSupplier(supplier);
  };

  if (loading && suppliers.length === 0) {
    return <LoadingSpinner message="Loading supply chain data..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Supply Chain Engagement
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage supplier questionnaires and track Scope 3 data collection
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PersonAdd />}
          onClick={() => setInviteOpen(true)}
        >
          Invite Supplier
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Engagement dashboard */}
        {summary && (
          <Grid item xs={12}>
            <EngagementDashboard summary={summary} />
          </Grid>
        )}

        {/* Hotspot map */}
        {summary && summary.hotspot_categories.length > 0 && (
          <Grid item xs={12} md={selectedSupplier ? 6 : 8}>
            <HotspotMap hotspots={summary.hotspot_categories} />
          </Grid>
        )}

        {/* Selected supplier tracker */}
        {selectedSupplier && (
          <Grid item xs={12} md={6}>
            <SupplierTracker supplier={selectedSupplier} />
          </Grid>
        )}

        {/* Supplier list */}
        <Grid item xs={12}>
          <SupplierList
            suppliers={suppliers}
            onSelect={handleSupplierSelect}
          />
        </Grid>
      </Grid>

      {/* Invite dialog */}
      <Dialog open={inviteOpen} onClose={() => setInviteOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Invite Supplier</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Supplier Name"
              value={inviteForm.supplier_name}
              onChange={(e) => setInviteForm({ ...inviteForm, supplier_name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={inviteForm.supplier_email}
              onChange={(e) => setInviteForm({ ...inviteForm, supplier_email: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Country"
              value={inviteForm.supplier_country}
              onChange={(e) => setInviteForm({ ...inviteForm, supplier_country: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Sector"
              value={inviteForm.supplier_sector}
              onChange={(e) => setInviteForm({ ...inviteForm, supplier_sector: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Custom Message (optional)"
              value={inviteForm.message}
              onChange={(e) => setInviteForm({ ...inviteForm, message: e.target.value })}
              fullWidth
              multiline
              rows={3}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInviteOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleInvite}
            disabled={!inviteForm.supplier_name || !inviteForm.supplier_email}
          >
            Send Invitation
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SupplyChainPage;
