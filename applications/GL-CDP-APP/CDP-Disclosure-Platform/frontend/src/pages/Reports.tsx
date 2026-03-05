/**
 * Reports Page - Report generation, preview, and submission
 *
 * Composes ReportBuilder, PreviewPanel, ExportDialog,
 * SubmissionChecklist, and VerificationPackage for the
 * complete report workflow.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
} from '@mui/material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  generateReport,
  fetchReports,
  fetchChecklist,
  submitToORS,
} from '../store/slices/reportsSlice';
import { fetchVerificationRecords, fetchVerificationSummary } from '../store/slices/verificationSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ReportBuilder from '../components/reports/ReportBuilder';
import PreviewPanel from '../components/reports/PreviewPanel';
import ExportDialog from '../components/reports/ExportDialog';
import SubmissionChecklistComponent from '../components/reports/SubmissionChecklist';
import VerificationPackage from '../components/reports/VerificationPackage';
import { ReportFormat } from '../types';

const DEMO_ORG_ID = 'demo-org';
const DEMO_QUESTIONNAIRE_ID = 'demo-questionnaire';

const ReportsPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { reports, checklist, generating, loading, error } = useAppSelector(
    (s) => s.reports,
  );
  const { data: dashboardData } = useAppSelector((s) => s.dashboard);
  const { result: scoringResult } = useAppSelector((s) => s.scoring);
  const { records: verificationRecords, summary: verificationSummary } =
    useAppSelector((s) => s.verification);

  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    dispatch(fetchReports(DEMO_QUESTIONNAIRE_ID));
    dispatch(fetchChecklist(DEMO_QUESTIONNAIRE_ID));
    dispatch(fetchVerificationRecords(DEMO_ORG_ID));
    dispatch(fetchVerificationSummary(DEMO_ORG_ID));
  }, [dispatch]);

  const handleGenerate = (format: ReportFormat, title: string) => {
    dispatch(generateReport({
      questionnaire_id: DEMO_QUESTIONNAIRE_ID,
      format,
      title,
    }));
  };

  const handleExport = (format: ReportFormat) => {
    dispatch(generateReport({
      questionnaire_id: DEMO_QUESTIONNAIRE_ID,
      format,
    }));
    setExportDialogOpen(false);
  };

  const handleDownload = (reportId: string) => {
    const report = reports.find((r) => r.id === reportId);
    if (report?.file_url) {
      window.open(report.file_url, '_blank');
    }
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      await dispatch(submitToORS(DEMO_QUESTIONNAIRE_ID)).unwrap();
    } finally {
      setSubmitting(false);
    }
  };

  if (loading && reports.length === 0 && !checklist) {
    return <LoadingSpinner message="Loading reports..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Reports & Submission
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Generate reports, review submission checklist, and submit to CDP ORS
      </Typography>

      <Grid container spacing={3}>
        {/* Report builder */}
        <Grid item xs={12} md={8}>
          <ReportBuilder
            reports={reports}
            generating={generating}
            onGenerate={handleGenerate}
            onDownload={handleDownload}
          />
        </Grid>

        {/* Preview panel */}
        <Grid item xs={12} md={4}>
          <PreviewPanel
            modules={dashboardData?.module_progress || []}
            scoringResult={scoringResult}
            reportingYear={dashboardData?.reporting_year || new Date().getFullYear()}
            orgName="Organization"
          />
        </Grid>

        {/* Submission checklist */}
        {checklist && (
          <Grid item xs={12} md={6}>
            <SubmissionChecklistComponent
              checklist={checklist}
              onSubmit={handleSubmit}
              submitting={submitting}
            />
          </Grid>
        )}

        {/* Verification package */}
        {verificationSummary && (
          <Grid item xs={12} md={6}>
            <VerificationPackage
              summary={verificationSummary}
              records={verificationRecords}
            />
          </Grid>
        )}
      </Grid>

      {/* Export dialog */}
      <ExportDialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        onExport={handleExport}
        exporting={generating}
      />
    </Box>
  );
};

export default ReportsPage;
