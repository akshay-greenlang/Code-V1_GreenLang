/**
 * SupplierManagement - Page for managing suppliers.
 *
 * Features "Add Supplier" and "Bulk Import" buttons, the SupplierTable
 * with full filtering, and a detail drawer showing SupplierDetail when
 * a row is clicked.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  Drawer,
  CircularProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import SupplierTable, { SupplierFilters } from '../components/suppliers/SupplierTable';
import SupplierDetail from '../components/suppliers/SupplierDetail';
import SupplierForm from '../components/suppliers/SupplierForm';
import apiClient from '../services/api';
import type { Supplier, SupplierCreateRequest, Plot, Document, DueDiligenceStatement } from '../types';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const SupplierManagement: React.FC = () => {
  // List state
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Detail drawer
  const [selectedSupplier, setSelectedSupplier] = useState<Supplier | null>(null);
  const [detailPlots, setDetailPlots] = useState<Plot[]>([]);
  const [detailDocs, setDetailDocs] = useState<Document[]>([]);
  const [detailDDS, setDetailDDS] = useState<DueDiligenceStatement[]>([]);

  // Form dialog
  const [formOpen, setFormOpen] = useState(false);
  const [editingSupplier, setEditingSupplier] = useState<Supplier | undefined>(undefined);
  const [formLoading, setFormLoading] = useState(false);

  // Snackbar
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // Filters
  const [filters, setFilters] = useState<SupplierFilters>({
    search: '',
    country: '',
    commodity: '',
    risk_level: '',
    compliance_status: '',
    sort_by: 'name',
    sort_order: 'asc',
  });

  // Fetch suppliers
  const fetchSuppliers = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getSuppliers({
        page: page + 1,
        per_page: rowsPerPage,
        search: filters.search || undefined,
        country: filters.country || undefined,
        commodity: (filters.commodity as Supplier['commodities'][0]) || undefined,
        risk_level: (filters.risk_level as Supplier['risk_level']) || undefined,
        compliance_status: (filters.compliance_status as Supplier['compliance_status']) || undefined,
        sort_by: filters.sort_by || undefined,
        sort_order: (filters.sort_order as 'asc' | 'desc') || undefined,
      });
      setSuppliers(result.items);
      setTotalCount(result.total);
    } catch {
      setError('Failed to load suppliers.');
    } finally {
      setLoading(false);
    }
  }, [page, rowsPerPage, filters]);

  useEffect(() => {
    fetchSuppliers();
  }, [fetchSuppliers]);

  // Load detail data when supplier selected
  useEffect(() => {
    if (!selectedSupplier) return;
    const id = selectedSupplier.id;

    Promise.all([
      apiClient.getPlots({ supplier_id: id, per_page: 100 }),
      apiClient.getDocuments({ supplier_id: id, per_page: 100 }),
      apiClient.getDDSList({ supplier_id: id, per_page: 100 }),
    ]).then(([plotsRes, docsRes, ddsRes]) => {
      setDetailPlots(plotsRes.items);
      setDetailDocs(docsRes.items);
      setDetailDDS(ddsRes.items);
    }).catch(() => {});
  }, [selectedSupplier]);

  // Handlers
  const handleRowClick = (supplier: Supplier) => {
    setSelectedSupplier(supplier);
  };

  const handleCloseDrawer = () => {
    setSelectedSupplier(null);
    setDetailPlots([]);
    setDetailDocs([]);
    setDetailDDS([]);
  };

  const handleAddSupplier = () => {
    setEditingSupplier(undefined);
    setFormOpen(true);
  };

  const handleEditSupplier = (supplier: Supplier) => {
    setEditingSupplier(supplier);
    setFormOpen(true);
  };

  const handleFormSubmit = async (data: SupplierCreateRequest) => {
    try {
      setFormLoading(true);
      if (editingSupplier) {
        await apiClient.updateSupplier(editingSupplier.id, data);
        setSnackbar({ open: true, message: 'Supplier updated successfully.', severity: 'success' });
      } else {
        await apiClient.createSupplier(data);
        setSnackbar({ open: true, message: 'Supplier created successfully.', severity: 'success' });
      }
      setFormOpen(false);
      fetchSuppliers();
    } catch {
      setSnackbar({ open: true, message: 'Failed to save supplier.', severity: 'error' });
    } finally {
      setFormLoading(false);
    }
  };

  const handleDeleteSupplier = async (supplier: Supplier) => {
    if (!window.confirm(`Delete supplier "${supplier.name}"? This cannot be undone.`)) return;
    try {
      await apiClient.deleteSupplier(supplier.id);
      setSnackbar({ open: true, message: 'Supplier deleted.', severity: 'success' });
      handleCloseDrawer();
      fetchSuppliers();
    } catch {
      setSnackbar({ open: true, message: 'Failed to delete supplier.', severity: 'error' });
    }
  };

  const handleGenerateDDS = (supplier: Supplier) => {
    // Navigate to DDS wizard -- in a full app this would use react-router
    setSnackbar({ open: true, message: `Opening DDS wizard for ${supplier.name}...`, severity: 'success' });
  };

  const handleRunPipeline = async (supplier: Supplier) => {
    try {
      await apiClient.startPipeline({
        supplier_id: supplier.id,
        commodity: supplier.commodities[0],
      });
      setSnackbar({ open: true, message: `Pipeline started for ${supplier.name}.`, severity: 'success' });
    } catch {
      setSnackbar({ open: true, message: 'Failed to start pipeline.', severity: 'error' });
    }
  };

  const handleBulkExport = (ids: string[]) => {
    setSnackbar({ open: true, message: `Exporting ${ids.length} suppliers...`, severity: 'success' });
  };

  const handleBulkGenerateDDS = (ids: string[]) => {
    setSnackbar({ open: true, message: `Generating DDS for ${ids.length} suppliers...`, severity: 'success' });
  };

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h4" fontWeight={700}>
          Supplier Management
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<UploadFileIcon />}>
            Bulk Import
          </Button>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleAddSupplier}>
            Add Supplier
          </Button>
        </Stack>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Supplier Table */}
      <SupplierTable
        suppliers={suppliers}
        totalCount={totalCount}
        page={page}
        rowsPerPage={rowsPerPage}
        loading={loading}
        onPageChange={setPage}
        onRowsPerPageChange={(rpp) => { setRowsPerPage(rpp); setPage(0); }}
        onRowClick={handleRowClick}
        onFilterChange={setFilters}
        onBulkExport={handleBulkExport}
        onBulkGenerateDDS={handleBulkGenerateDDS}
      />

      {/* Detail Drawer */}
      <Drawer
        anchor="right"
        open={Boolean(selectedSupplier)}
        onClose={handleCloseDrawer}
        PaperProps={{ sx: { width: { xs: '100%', md: 720 }, p: 3 } }}
      >
        {selectedSupplier && (
          <SupplierDetail
            supplier={selectedSupplier}
            plots={detailPlots}
            documents={detailDocs}
            ddsList={detailDDS}
            onEdit={handleEditSupplier}
            onGenerateDDS={handleGenerateDDS}
            onRunPipeline={handleRunPipeline}
            onDelete={handleDeleteSupplier}
          />
        )}
      </Drawer>

      {/* Add/Edit Dialog */}
      <Dialog
        open={formOpen}
        onClose={() => setFormOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogContent sx={{ p: 0 }}>
          <SupplierForm
            supplier={editingSupplier}
            onSubmit={handleFormSubmit}
            onCancel={() => setFormOpen(false)}
            loading={formLoading}
          />
        </DialogContent>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SupplierManagement;
