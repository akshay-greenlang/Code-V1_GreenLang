/**
 * DDSManagement - Page for managing Due Diligence Statements.
 *
 * Stats bar with status counts, "Generate DDS" and "Bulk Generate" buttons,
 * DDSTable with filters, detail drawer showing DDSDetail, and a
 * DDSValidation panel.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Chip,
  Drawer,
  Dialog,
  DialogContent,
  CircularProgress,
  Alert,
  Snackbar,
  Paper,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PlaylistAddCheckIcon from '@mui/icons-material/PlaylistAddCheck';
import DDSTable, { DDSFilters } from '../components/dds/DDSTable';
import DDSDetail from '../components/dds/DDSDetail';
import DDSWizard from '../components/dds/DDSWizard';
import DDSValidation from '../components/dds/DDSValidation';
import apiClient from '../services/api';
import type {
  DueDiligenceStatement,
  DDSStatus,
  DDSValidationResult,
  Supplier,
  Plot,
  DDSGenerateRequest,
  DocumentGapAnalysis,
} from '../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_CHIP_COLORS: Record<string, { bg: string; color: string }> = {
  draft: { bg: '#e0e0e0', color: '#424242' },
  pending_review: { bg: '#bbdefb', color: '#1565c0' },
  validated: { bg: '#b2dfdb', color: '#00695c' },
  submitted: { bg: '#ffe0b2', color: '#e65100' },
  accepted: { bg: '#c8e6c9', color: '#2e7d32' },
  rejected: { bg: '#ffcdd2', color: '#c62828' },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DDSManagement: React.FC = () => {
  // List state
  const [ddsList, setDDSList] = useState<DueDiligenceStatement[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [filters, setFilters] = useState<DDSFilters>({
    search: '',
    status: '',
    commodity: '',
    year: '',
  });

  // Detail drawer
  const [selectedDDS, setSelectedDDS] = useState<DueDiligenceStatement | null>(null);
  const [validation, setValidation] = useState<DDSValidationResult | null>(null);

  // Wizard dialog
  const [wizardOpen, setWizardOpen] = useState(false);
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [wizardLoading, setWizardLoading] = useState(false);

  // Snackbar
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false, message: '', severity: 'success',
  });

  // Status counts
  const statusCounts = ddsList.reduce<Record<string, number>>((acc, dds) => {
    acc[dds.status] = (acc[dds.status] || 0) + 1;
    return acc;
  }, {});

  // Fetch DDS list
  const fetchDDS = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getDDSList({
        page: page + 1,
        per_page: rowsPerPage,
        search: filters.search || undefined,
        status: (filters.status as DDSStatus) || undefined,
        commodity: filters.commodity ? (filters.commodity as DueDiligenceStatement['commodity']) : undefined,
        sort_by: 'created_at',
        sort_order: 'desc',
      });
      setDDSList(result.items);
      setTotalCount(result.total);
    } catch {
      setError('Failed to load DDS records.');
    } finally {
      setLoading(false);
    }
  }, [page, rowsPerPage, filters]);

  useEffect(() => {
    fetchDDS();
  }, [fetchDDS]);

  // Fetch suppliers for wizard
  useEffect(() => {
    apiClient
      .getSuppliers({ per_page: 500 })
      .then((res) => setSuppliers(res.items))
      .catch(() => {});
  }, []);

  // Handlers
  const handleView = (dds: DueDiligenceStatement) => {
    setSelectedDDS(dds);
    setValidation(null);
  };

  const handleValidate = async (dds: DueDiligenceStatement) => {
    try {
      const result = await apiClient.validateDDS(dds.id);
      setValidation(result);
      setSelectedDDS(dds);
      setSnackbar({ open: true, message: 'Validation complete.', severity: 'success' });
    } catch {
      setSnackbar({ open: true, message: 'Validation failed.', severity: 'error' });
    }
  };

  const handleSubmit = async (dds: DueDiligenceStatement) => {
    try {
      await apiClient.submitDDS(dds.id);
      setSnackbar({ open: true, message: 'DDS submitted successfully.', severity: 'success' });
      fetchDDS();
    } catch {
      setSnackbar({ open: true, message: 'Submission failed.', severity: 'error' });
    }
  };

  const handleDownload = async (dds: DueDiligenceStatement, format: 'pdf' | 'xml' | 'json') => {
    try {
      const blob = await apiClient.downloadDDS(dds.id, format);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `DDS_${dds.reference_number}.${format}`;
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      setSnackbar({ open: true, message: 'Download failed.', severity: 'error' });
    }
  };

  const handleAmend = async (dds: DueDiligenceStatement) => {
    setSnackbar({ open: true, message: `Amending DDS ${dds.reference_number}...`, severity: 'success' });
  };

  const handleGenerate = async (request: DDSGenerateRequest) => {
    try {
      setWizardLoading(true);
      await apiClient.generateDDS(request);
      setSnackbar({ open: true, message: 'DDS generated successfully.', severity: 'success' });
      setWizardOpen(false);
      fetchDDS();
    } catch {
      setSnackbar({ open: true, message: 'DDS generation failed.', severity: 'error' });
    } finally {
      setWizardLoading(false);
    }
  };

  const handleFetchPlots = async (supplierId: string): Promise<Plot[]> => {
    const res = await apiClient.getPlots({ supplier_id: supplierId, per_page: 200 });
    return res.items;
  };

  const handleFetchDocGap = async (supplierId: string): Promise<DocumentGapAnalysis> => {
    return apiClient.getDocumentGapAnalysis(supplierId);
  };

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h4" fontWeight={700}>
          DDS Management
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<PlaylistAddCheckIcon />}>
            Bulk Generate
          </Button>
          <Button variant="contained" startIcon={<AddIcon />} onClick={() => setWizardOpen(true)}>
            Generate DDS
          </Button>
        </Stack>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Status Chips Bar */}
      <Paper sx={{ p: 1.5, mb: 2 }}>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {(['draft', 'pending_review', 'validated', 'submitted', 'accepted', 'rejected'] as DDSStatus[]).map(
            (status) => {
              const chipStyle = STATUS_CHIP_COLORS[status] ?? STATUS_CHIP_COLORS.draft;
              const count = statusCounts[status] ?? 0;
              return (
                <Chip
                  key={status}
                  label={`${status.replace('_', ' ')}: ${count}`}
                  sx={{
                    backgroundColor: chipStyle.bg,
                    color: chipStyle.color,
                    fontWeight: 600,
                    textTransform: 'capitalize',
                    cursor: 'pointer',
                  }}
                  onClick={() => setFilters((f) => ({ ...f, status: f.status === status ? '' : status }))}
                  variant={filters.status === status ? 'filled' : 'outlined'}
                />
              );
            }
          )}
        </Stack>
      </Paper>

      {/* DDS Table */}
      <DDSTable
        ddsList={ddsList}
        totalCount={totalCount}
        page={page}
        rowsPerPage={rowsPerPage}
        loading={loading}
        onPageChange={setPage}
        onRowsPerPageChange={(rpp) => { setRowsPerPage(rpp); setPage(0); }}
        onView={handleView}
        onValidate={handleValidate}
        onSubmit={handleSubmit}
        onDownload={handleDownload}
        onAmend={handleAmend}
        onFilterChange={setFilters}
      />

      {/* Detail Drawer */}
      <Drawer
        anchor="right"
        open={Boolean(selectedDDS)}
        onClose={() => { setSelectedDDS(null); setValidation(null); }}
        PaperProps={{ sx: { width: { xs: '100%', md: 700 }, p: 3 } }}
      >
        {selectedDDS && (
          <Box>
            <DDSDetail
              dds={selectedDDS}
              onValidate={handleValidate}
              onSubmit={handleSubmit}
              onDownload={handleDownload}
              onAmend={handleAmend}
            />

            {/* Validation Panel */}
            {validation && (
              <Box mt={3}>
                <Divider sx={{ mb: 2 }} />
                <DDSValidation validation={validation} />
              </Box>
            )}
          </Box>
        )}
      </Drawer>

      {/* DDS Wizard Dialog */}
      <Dialog
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogContent sx={{ p: 0 }}>
          <DDSWizard
            suppliers={suppliers}
            onFetchPlots={handleFetchPlots}
            onFetchDocGap={handleFetchDocGap}
            onGenerate={handleGenerate}
            onCancel={() => setWizardOpen(false)}
            loading={wizardLoading}
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
        <Alert onClose={() => setSnackbar((s) => ({ ...s, open: false }))} severity={snackbar.severity} variant="filled">
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DDSManagement;
