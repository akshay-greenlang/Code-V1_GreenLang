import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Chip,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import { GridColDef } from '@mui/x-data-grid';
import { Add, Download, Delete, Refresh } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchReports, generateReport, downloadReport, deleteReport, setPage, setPageSize } from '../store/slices/reportsSlice';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { formatDate, getStatusColor } from '../utils/formatters';
import type { ReportRequest, Report } from '../types';

const Reports: React.FC = () => {
  const dispatch = useAppDispatch();
  const { reports, pagination, loading, generating, error } = useAppSelector((state) => state.reports);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [reportRequest, setReportRequest] = useState<Partial<ReportRequest>>({
    type: 'esrs_e1',
    format: 'pdf',
    startDate: '',
    endDate: '',
  });

  useEffect(() => {
    dispatch(fetchReports({ page: pagination.page, pageSize: pagination.pageSize }));
  }, [dispatch, pagination.page, pagination.pageSize]);

  const handleGenerateReport = async () => {
    if (reportRequest.type && reportRequest.format && reportRequest.startDate && reportRequest.endDate) {
      await dispatch(generateReport(reportRequest as ReportRequest));
      setDialogOpen(false);
      setReportRequest({
        type: 'esrs_e1',
        format: 'pdf',
        startDate: '',
        endDate: '',
      });
    }
  };

  const handleDownload = (reportId: string) => {
    dispatch(downloadReport(reportId));
  };

  const handleDelete = async (reportId: string) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      await dispatch(deleteReport(reportId));
    }
  };

  const getReportTypeLabel = (type: string): string => {
    const typeMap: Record<string, string> = {
      esrs_e1: 'ESRS E1 (EU CSRD)',
      cdp: 'CDP Questionnaire',
      ghg_protocol: 'GHG Protocol',
      iso_14083: 'ISO 14083',
      ifrs_s2: 'IFRS S2',
    };
    return typeMap[type] || type;
  };

  const columns: GridColDef[] = [
    {
      field: 'name',
      headerName: 'Report Name',
      flex: 1,
      minWidth: 250,
    },
    {
      field: 'type',
      headerName: 'Type',
      width: 180,
      renderCell: (params) => getReportTypeLabel(params.value),
    },
    {
      field: 'reportingPeriod',
      headerName: 'Period',
      width: 200,
      renderCell: (params) =>
        `${formatDate(params.value.startDate)} - ${formatDate(params.value.endDate)}`,
    },
    {
      field: 'fileFormat',
      headerName: 'Format',
      width: 100,
      renderCell: (params) => params.value.toUpperCase(),
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 130,
      renderCell: (params) => (
        <Chip label={params.value} size="small" color={getStatusColor(params.value)} />
      ),
    },
    {
      field: 'createdAt',
      headerName: 'Created',
      width: 150,
      renderCell: (params) => formatDate(params.value),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 150,
      sortable: false,
      renderCell: (params) => (
        <Box>
          <Tooltip title="Download">
            <IconButton
              size="small"
              onClick={() => handleDownload(params.row.id)}
              disabled={params.row.status !== 'completed'}
            >
              <Download />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton size="small" onClick={() => handleDelete(params.row.id)} color="error">
              <Delete />
            </IconButton>
          </Tooltip>
        </Box>
      ),
    },
  ];

  if (loading && reports.length === 0) {
    return <LoadingSpinner message="Loading reports..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Reports</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => dispatch(fetchReports())}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setDialogOpen(true)}
          >
            Generate Report
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {generating && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Generating report... This may take a few minutes.
        </Alert>
      )}

      <DataTable
        rows={reports}
        columns={columns}
        loading={loading}
        pageSize={pagination.pageSize}
        page={pagination.page - 1}
        totalRows={pagination.total}
        onPageChange={(page) => dispatch(setPage(page + 1))}
        onPageSizeChange={(pageSize) => dispatch(setPageSize(pageSize))}
      />

      {/* Generate Report Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Generate New Report</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Report Type</InputLabel>
                <Select
                  value={reportRequest.type}
                  label="Report Type"
                  onChange={(e) =>
                    setReportRequest({ ...reportRequest, type: e.target.value as Report['type'] })
                  }
                >
                  <MenuItem value="esrs_e1">ESRS E1 (EU CSRD)</MenuItem>
                  <MenuItem value="cdp">CDP Questionnaire</MenuItem>
                  <MenuItem value="ghg_protocol">GHG Protocol</MenuItem>
                  <MenuItem value="iso_14083">ISO 14083</MenuItem>
                  <MenuItem value="ifrs_s2">IFRS S2</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>File Format</InputLabel>
                <Select
                  value={reportRequest.format}
                  label="File Format"
                  onChange={(e) =>
                    setReportRequest({ ...reportRequest, format: e.target.value as Report['fileFormat'] })
                  }
                >
                  <MenuItem value="pdf">PDF</MenuItem>
                  <MenuItem value="excel">Excel (XLSX)</MenuItem>
                  <MenuItem value="json">JSON</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Start Date"
                type="date"
                value={reportRequest.startDate}
                onChange={(e) => setReportRequest({ ...reportRequest, startDate: e.target.value })}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="End Date"
                type="date"
                value={reportRequest.endDate}
                onChange={(e) => setReportRequest({ ...reportRequest, endDate: e.target.value })}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            <Grid item xs={12}>
              <Alert severity="info">
                The report will be generated based on all transactions within the selected date range.
                Generation may take 1-5 minutes depending on data volume.
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleGenerateReport}
            variant="contained"
            disabled={!reportRequest.type || !reportRequest.startDate || !reportRequest.endDate}
          >
            Generate
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Reports;
