import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Alert,
  Paper,
  Grid,
} from '@mui/material';
import { GridColDef, GridRowParams } from '@mui/x-data-grid';
import { Send, Refresh } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchSuppliers, createCampaign, setPage, setPageSize, setFilters } from '../store/slices/suppliersSlice';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { formatCurrency, formatNumber, formatEmissions, formatRelativeTime, getStatusColor } from '../utils/formatters';
import type { Supplier } from '../types';

const SupplierManagement: React.FC = () => {
  const dispatch = useAppDispatch();
  const { suppliers, pagination, filters, loading, error } = useAppSelector((state) => state.suppliers);
  const [selectedSuppliers, setSelectedSuppliers] = useState<string[]>([]);
  const [campaignDialogOpen, setCampaignDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    dispatch(fetchSuppliers({
      page: pagination.page,
      pageSize: pagination.pageSize,
      search: filters.search,
      status: filters.status,
    }));
  }, [dispatch, pagination.page, pagination.pageSize, filters]);

  const handleSearch = () => {
    dispatch(setFilters({ ...filters, search: searchQuery }));
  };

  const handleRowClick = (params: GridRowParams) => {
    const supplierId = params.row.id;
    setSelectedSuppliers((prev) =>
      prev.includes(supplierId)
        ? prev.filter((id) => id !== supplierId)
        : [...prev, supplierId]
    );
  };

  const handleCreateCampaign = async () => {
    if (selectedSuppliers.length > 0) {
      await dispatch(createCampaign({ supplierIds: selectedSuppliers }));
      setCampaignDialogOpen(false);
      setSelectedSuppliers([]);
    }
  };

  const columns: GridColDef[] = [
    {
      field: 'name',
      headerName: 'Supplier Name',
      flex: 1,
      minWidth: 200,
    },
    {
      field: 'supplierId',
      headerName: 'Supplier ID',
      width: 150,
    },
    {
      field: 'country',
      headerName: 'Country',
      width: 120,
    },
    {
      field: 'totalEmissionsKgCO2e',
      headerName: 'Emissions',
      width: 150,
      renderCell: (params) => formatEmissions(params.value),
    },
    {
      field: 'totalSpendUsd',
      headerName: 'Total Spend',
      width: 130,
      renderCell: (params) => formatCurrency(params.value),
    },
    {
      field: 'dataQuality',
      headerName: 'Data Quality',
      width: 120,
      renderCell: (params) => (
        <Chip
          label={params.value.toUpperCase()}
          size="small"
          color={params.value === 'tier1' ? 'success' : params.value === 'tier2' ? 'warning' : 'default'}
        />
      ),
    },
    {
      field: 'responseRate',
      headerName: 'Response Rate',
      width: 130,
      renderCell: (params) => `${(params.value * 100).toFixed(0)}%`,
    },
    {
      field: 'pcfCount',
      headerName: 'PCF Count',
      width: 100,
      align: 'center',
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => (
        <Chip label={params.value} size="small" color={getStatusColor(params.value)} />
      ),
    },
  ];

  if (loading && suppliers.length === 0) {
    return <LoadingSpinner message="Loading suppliers..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Supplier Management</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => dispatch(fetchSuppliers())}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Send />}
            onClick={() => setCampaignDialogOpen(true)}
            disabled={selectedSuppliers.length === 0}
          >
            Create Campaign ({selectedSuppliers.length})
          </Button>
        </Box>
      </Box>

      {/* Search and Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              label="Search Suppliers"
              placeholder="Search by name, ID, or country..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Button variant="contained" fullWidth onClick={handleSearch}>
              Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {selectedSuppliers.length > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {selectedSuppliers.length} supplier(s) selected for campaign
        </Alert>
      )}

      <DataTable
        rows={suppliers}
        columns={columns}
        loading={loading}
        pageSize={pagination.pageSize}
        page={pagination.page - 1}
        totalRows={pagination.total}
        onPageChange={(page) => dispatch(setPage(page + 1))}
        onPageSizeChange={(pageSize) => dispatch(setPageSize(pageSize))}
        onRowClick={handleRowClick}
      />

      {/* Create Campaign Dialog */}
      <Dialog
        open={campaignDialogOpen}
        onClose={() => setCampaignDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create Engagement Campaign</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            You are about to create an engagement campaign for {selectedSuppliers.length} supplier(s).
            Suppliers will receive an email invitation to submit their Product Carbon Footprint (PCF) data.
          </Alert>
          <Typography variant="body2" paragraph>
            The campaign will include:
          </Typography>
          <ul>
            <li>Invitation email with secure portal link</li>
            <li>PCF submission guidelines and templates</li>
            <li>Follow-up reminders (4-touch sequence)</li>
            <li>Progress tracking and analytics</li>
          </ul>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCampaignDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateCampaign} variant="contained" color="primary">
            Create Campaign
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SupplierManagement;
