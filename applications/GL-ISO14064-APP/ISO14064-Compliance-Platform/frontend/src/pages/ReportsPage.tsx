/**
 * ReportsPage - Report generation, mandatory elements, compliance gauge,
 * and past reports listing.
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Alert,
  Grid,
  Button,
  Card,
  CardContent,
} from '@mui/material';
import { NoteAdd, Download } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchMandatoryElements,
  fetchReports,
  generateReport,
  downloadReport,
} from '../store/slices/reportsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import MandatoryElementsChecklist from '../components/reports/MandatoryElementsChecklist';
import ComplianceGauge from '../components/reports/ComplianceGauge';
import ReportGenerator from '../components/reports/ReportGenerator';
import DataTable, { Column } from '../components/common/DataTable';
import type { ISOReport, GenerateReportRequest } from '../types';
import { formatDate } from '../utils/formatters';

const DEMO_INVENTORY_ID = 'demo-inventory';

const ReportsPage: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { reports, mandatoryElements, completeness_pct, generating, loading, error } =
    useSelector((s: AppRootState) => s.reports);
  const [generatorOpen, setGeneratorOpen] = useState(false);

  useEffect(() => {
    dispatch(fetchMandatoryElements(DEMO_INVENTORY_ID));
    dispatch(fetchReports(DEMO_INVENTORY_ID));
  }, [dispatch]);

  const handleGenerate = (data: GenerateReportRequest) => {
    dispatch(generateReport(data));
  };

  const handleDownload = (reportId: string) => {
    dispatch(downloadReport(reportId));
  };

  const reportColumns: Column<ISOReport>[] = [
    { id: 'title', label: 'Title' },
    { id: 'reporting_year', label: 'Year' },
    {
      id: 'format',
      label: 'Format',
      render: (row) => row.format.toUpperCase(),
    },
    {
      id: 'mandatory_completeness_pct',
      label: 'Completeness',
      align: 'right',
      render: (row) => `${row.mandatory_completeness_pct.toFixed(0)}%`,
    },
    {
      id: 'generated_at',
      label: 'Generated',
      render: (row) => formatDate(row.generated_at),
    },
    {
      id: 'actions',
      label: 'Actions',
      sortable: false,
      align: 'center',
      render: (row) => (
        <Button
          size="small"
          startIcon={<Download />}
          onClick={() => handleDownload(row.id)}
        >
          Download
        </Button>
      ),
    },
  ];

  if (loading && reports.length === 0 && mandatoryElements.length === 0) {
    return <LoadingSpinner message="Loading reports..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            ISO 14064-1 Reports
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Report generation, mandatory element compliance, and export
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<NoteAdd />}
          onClick={() => setGeneratorOpen(true)}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          Generate Report
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Compliance Gauge */}
        <Grid item xs={12} md={4}>
          <ComplianceGauge completeness_pct={completeness_pct} />
        </Grid>

        {/* Mandatory Elements */}
        <Grid item xs={12} md={8}>
          <MandatoryElementsChecklist
            elements={mandatoryElements}
            completeness_pct={completeness_pct}
          />
        </Grid>
      </Grid>

      {/* Past Reports */}
      <Typography variant="h6" fontWeight={600} gutterBottom>
        Report History
      </Typography>
      <DataTable
        columns={reportColumns}
        rows={reports}
        rowKey={(r) => r.id}
        searchPlaceholder="Search reports..."
      />

      {/* Generator Dialog */}
      <ReportGenerator
        open={generatorOpen}
        onClose={() => setGeneratorOpen(false)}
        onGenerate={handleGenerate}
        onDownload={handleDownload}
        inventoryId={DEMO_INVENTORY_ID}
        pastReports={reports}
        generating={generating}
      />
    </Box>
  );
};

export default ReportsPage;
