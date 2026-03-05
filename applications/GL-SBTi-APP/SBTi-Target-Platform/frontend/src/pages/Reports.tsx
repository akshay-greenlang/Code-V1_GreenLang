/**
 * Reports - Report generation, preview, export, and submission management.
 *
 * Generate target summaries, progress reports, validation reports,
 * SBTi submission packages, and annual disclosures.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Button, Chip, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  IconButton, Dialog, DialogTitle, DialogContent, DialogActions,
} from '@mui/material';
import { Download, Delete, Visibility, Description, PictureAsPdf } from '@mui/icons-material';
import ReportBuilder from '../components/reports/ReportBuilder';
import PreviewPanel from '../components/reports/PreviewPanel';
import ExportDialog from '../components/reports/ExportDialog';
import SubmissionPreview from '../components/reports/SubmissionPreview';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchReports, generateReport, selectReports, selectReportLoading, selectReportGenerating } from '../store/slices/reportSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_REPORTS = [
  { id: 'rpt_1', report_type: 'target_summary', title: 'Target Summary Report 2025', status: 'completed' as const, format: 'pdf', generated_at: '2025-02-28T14:30:00', file_size: 2456000, pages: 18 },
  { id: 'rpt_2', report_type: 'progress_report', title: 'Annual Progress Report FY2024', status: 'completed' as const, format: 'pdf', generated_at: '2025-02-25T10:15:00', file_size: 3120000, pages: 24 },
  { id: 'rpt_3', report_type: 'validation_report', title: 'Validation Report - Near-term Targets', status: 'completed' as const, format: 'pdf', generated_at: '2025-02-20T16:45:00', file_size: 1890000, pages: 12 },
  { id: 'rpt_4', report_type: 'submission_package', title: 'SBTi Submission Package v2', status: 'draft' as const, format: 'xlsx', generated_at: '2025-02-15T09:00:00', file_size: 4500000, pages: null },
  { id: 'rpt_5', report_type: 'annual_disclosure', title: 'Annual Disclosure 2024', status: 'generating' as const, format: 'pdf', generated_at: null, file_size: null, pages: null },
];

const TYPE_LABELS: Record<string, string> = {
  target_summary: 'Target Summary',
  progress_report: 'Progress Report',
  validation_report: 'Validation Report',
  submission_package: 'Submission Package',
  annual_disclosure: 'Annual Disclosure',
};

const STATUS_COLORS: Record<string, 'success' | 'warning' | 'info' | 'default'> = {
  completed: 'success',
  draft: 'warning',
  generating: 'info',
  failed: 'default',
};

const Reports: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const storeReports = useAppSelector(selectReports);
  const loading = useAppSelector(selectReportLoading);
  const generating = useAppSelector(selectReportGenerating);
  const [exportOpen, setExportOpen] = useState(false);
  const [previewId, setPreviewId] = useState<string | null>(null);

  useEffect(() => {
    dispatch(fetchReports(orgId));
  }, [dispatch, orgId]);

  const reports = storeReports.length > 0 ? storeReports : DEMO_REPORTS;

  const handleGenerate = (config: { report_type: string; target_ids: string[]; year: number }) => {
    dispatch(generateReport({ org_id: orgId, ...config }));
  };

  const formatFileSize = (bytes: number | null) => {
    if (!bytes) return '--';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (loading && storeReports.length === 0) return <LoadingSpinner message="Loading reports..." />;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Reports & Exports</Typography>
        <Typography variant="body2" color="text.secondary">
          Generate, preview, and export SBTi reports and submission packages
        </Typography>
      </Box>

      {/* Report Builder */}
      <Box sx={{ mb: 3 }}>
        <ReportBuilder onGenerate={handleGenerate} generating={generating} />
      </Box>

      {/* KPI Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Description color="primary" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{reports.length}</Typography>
              <Typography variant="body2" color="text.secondary">Total Reports</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <PictureAsPdf color="success" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>
                {reports.filter((r: any) => r.status === 'completed').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">Completed</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 700, mt: 1 }}>
                {reports.filter((r: any) => r.report_type === 'submission_package').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">Submission Packages</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 700, mt: 1 }}>
                {reports.filter((r: any) => r.status === 'generating').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">In Progress</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Reports Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Generated Reports</Typography>
            <Button variant="outlined" size="small" startIcon={<Download />} onClick={() => setExportOpen(true)}>
              Bulk Export
            </Button>
          </Box>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Report</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell>Format</TableCell>
                  <TableCell align="right">Size</TableCell>
                  <TableCell>Generated</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {reports.map((report: any) => (
                  <TableRow key={report.id} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{report.title}</TableCell>
                    <TableCell>
                      <Chip label={TYPE_LABELS[report.report_type] || report.report_type} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={report.status}
                        size="small"
                        color={STATUS_COLORS[report.status] || 'default'}
                        sx={{ textTransform: 'capitalize' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip label={report.format?.toUpperCase() || '--'} size="small" sx={{ fontSize: '0.65rem' }} />
                    </TableCell>
                    <TableCell align="right" sx={{ fontSize: '0.85rem' }}>
                      {formatFileSize(report.file_size)}
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.85rem' }}>
                      {report.generated_at ? new Date(report.generated_at).toLocaleString() : 'Pending...'}
                    </TableCell>
                    <TableCell align="center">
                      <IconButton
                        size="small"
                        onClick={() => setPreviewId(report.id)}
                        disabled={report.status !== 'completed'}
                      >
                        <Visibility fontSize="small" />
                      </IconButton>
                      <IconButton size="small" disabled={report.status !== 'completed'}>
                        <Download fontSize="small" />
                      </IconButton>
                      <IconButton size="small" color="error">
                        <Delete fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Export Dialog */}
      <ExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        onExport={(format) => { setExportOpen(false); }}
      />
    </Box>
  );
};

export default Reports;
