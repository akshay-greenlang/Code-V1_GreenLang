/**
 * InventoryList Page - List of ISO 14064-1 inventories
 *
 * Displays all inventories with status, year, totals, and navigation
 * to inventory detail. Supports creating new inventories.
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid,
  MenuItem,
  Chip,
} from '@mui/material';
import { Add } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import { fetchInventories, createInventory } from '../store/slices/inventorySlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import DataTable, { Column } from '../components/common/DataTable';
import StatusChip from '../components/common/StatusChip';
import type { ISOInventory, CreateInventoryRequest } from '../types';
import { ConsolidationApproach, GWPSource } from '../types';
import { formatDate } from '../utils/formatters';

const DEMO_ORG_ID = 'demo-org';

const InventoryList: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const navigate = useNavigate();
  const { inventories, loading, error } = useSelector(
    (s: AppRootState) => s.inventory,
  );
  const [createOpen, setCreateOpen] = useState(false);
  const [newInv, setNewInv] = useState<CreateInventoryRequest>({
    org_id: DEMO_ORG_ID,
    reporting_year: new Date().getFullYear() - 1,
    consolidation_approach: ConsolidationApproach.OPERATIONAL_CONTROL,
    gwp_source: GWPSource.AR6,
  });

  useEffect(() => {
    dispatch(fetchInventories(DEMO_ORG_ID));
  }, [dispatch]);

  const handleCreate = () => {
    dispatch(createInventory(newInv));
    setCreateOpen(false);
  };

  const columns: Column<ISOInventory>[] = [
    {
      id: 'reporting_year',
      label: 'Year',
      render: (row) => (
        <Typography variant="body2" fontWeight={600}>
          {row.reporting_year}
        </Typography>
      ),
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => <StatusChip status={row.status} />,
    },
    {
      id: 'consolidation_approach',
      label: 'Approach',
      render: (row) => (
        <Chip
          label={row.consolidation_approach.replace(/_/g, ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    {
      id: 'gwp_source',
      label: 'GWP Source',
      render: (row) => row.gwp_source.toUpperCase(),
    },
    {
      id: 'created_at',
      label: 'Created',
      render: (row) => formatDate(row.created_at),
    },
    {
      id: 'updated_at',
      label: 'Last Updated',
      render: (row) => formatDate(row.updated_at),
    },
  ];

  if (loading && inventories.length === 0) {
    return <LoadingSpinner message="Loading inventories..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            GHG Inventories
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ISO 14064-1 inventory management by reporting year
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateOpen(true)}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          New Inventory
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <DataTable
        columns={columns}
        rows={inventories}
        rowKey={(r) => r.id}
        onRowClick={(row) => navigate(`/inventories/${row.id}`)}
        searchPlaceholder="Search inventories..."
      />

      {/* Create dialog */}
      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Inventory</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Reporting Year"
                value={newInv.reporting_year}
                onChange={(e) =>
                  setNewInv({ ...newInv, reporting_year: Number(e.target.value) })
                }
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Consolidation Approach"
                value={newInv.consolidation_approach}
                onChange={(e) =>
                  setNewInv({
                    ...newInv,
                    consolidation_approach: e.target.value as ConsolidationApproach,
                  })
                }
              >
                {Object.values(ConsolidationApproach).map((a) => (
                  <MenuItem key={a} value={a}>
                    {a.replace(/_/g, ' ')}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="GWP Source"
                value={newInv.gwp_source}
                onChange={(e) =>
                  setNewInv({ ...newInv, gwp_source: e.target.value as GWPSource })
                }
              >
                {Object.values(GWPSource).map((g) => (
                  <MenuItem key={g} value={g}>
                    {g.toUpperCase()}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreate}
            variant="contained"
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default InventoryList;
