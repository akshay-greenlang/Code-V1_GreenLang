/**
 * Reports Page - Report generation and disclosure management
 *
 * Composes ReportBuilder configuration panel, ReportPreview section,
 * report history table, completeness checker results, and a
 * disclosure checklist view.
 */

import React, { useEffect, useState } from 'react';
import { Box, Typography, Alert, Tabs, Tab } from '@mui/material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import {
  generateReport,
  fetchReports,
  fetchDisclosures,
  fetchCompleteness,
  downloadReport,
} from '../store/slices/reportsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ReportBuilder from '../components/reports/ReportBuilder';
import ReportPreview from '../components/reports/ReportPreview';
import { ReportFormat, REPORT_SECTIONS } from '../types';
import type { GenerateReportRequest } from '../types';

const DEMO_INVENTORY_ID = 'demo-inventory';

const ReportsPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { reports, disclosures, completeness, generating, loading, error } = useAppSelector(
    (state) => state.reports
  );
  const [tabIndex, setTabIndex] = useState(0);
  const [selectedSections, setSelectedSections] = useState<string[]>(
    REPORT_SECTIONS.filter((s) => s.required).map((s) => s.id)
  );

  useEffect(() => {
    dispatch(fetchReports(DEMO_INVENTORY_ID));
    dispatch(fetchDisclosures(DEMO_INVENTORY_ID));
    dispatch(fetchCompleteness(DEMO_INVENTORY_ID));
  }, [dispatch]);

  const handleGenerate = (format: ReportFormat, sections: string[]) => {
    setSelectedSections(sections);
    const payload: GenerateReportRequest = {
      inventory_id: DEMO_INVENTORY_ID,
      format,
      title: `GHG Report - ${format.toUpperCase()}`,
      includes_scope1: true,
      includes_scope2: true,
      includes_scope3: true,
      includes_verification: sections.includes('verification'),
    };
    dispatch(generateReport(payload));
  };

  const handleDownload = (reportId: string) => {
    dispatch(downloadReport(reportId));
  };

  if (loading && reports.length === 0) return <LoadingSpinner message="Loading reports..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Tabs value={tabIndex} onChange={(_, v) => setTabIndex(v)} sx={{ mb: 3 }}>
        <Tab label="Generate Report" />
        <Tab label="Preview & Disclosure" />
      </Tabs>

      {tabIndex === 0 && (
        <ReportBuilder
          inventoryId={DEMO_INVENTORY_ID}
          reports={reports}
          generating={generating}
          onGenerate={handleGenerate}
          onDownload={handleDownload}
        />
      )}

      {tabIndex === 1 && (
        <ReportPreview
          selectedSections={selectedSections}
          completeness={completeness}
          disclosures={disclosures}
        />
      )}
    </Box>
  );
};

export default ReportsPage;
