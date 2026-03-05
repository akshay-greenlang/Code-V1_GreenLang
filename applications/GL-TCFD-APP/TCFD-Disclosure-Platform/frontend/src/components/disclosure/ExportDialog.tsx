/**
 * ExportDialog - Export format selector with configuration options for TCFD disclosures.
 *
 * Supports PDF, DOCX, Excel, JSON, and XBRL export formats with pillar/section selection,
 * watermark and branding options, and metadata inclusion toggles.
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  FormControlLabel,
  Checkbox,
  Radio,
  RadioGroup,
  FormControl,
  FormLabel,
  Divider,
  Chip,
  LinearProgress,
  Alert,
  Grid,
  TextField,
} from '@mui/material';
import {
  PictureAsPdf,
  Description,
  TableChart,
  Code,
  DataObject,
  Download,
  CheckCircle,
} from '@mui/icons-material';
import type { ExportFormat, TCFDPillar } from '../../types';

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
  onExport: (format: ExportFormat, options: ExportOptions) => void;
  isExporting?: boolean;
  exportProgress?: number;
}

interface ExportOptions {
  format: ExportFormat;
  pillars: TCFDPillar[];
  includeCharts: boolean;
  includeEvidence: boolean;
  includeComplianceMatrix: boolean;
  includeAppendix: boolean;
  watermark: boolean;
  draft: boolean;
  reportTitle: string;
  organizationName: string;
  reportingPeriod: string;
}

const FORMAT_CONFIG: Record<ExportFormat, { label: string; icon: React.ReactNode; description: string }> = {
  pdf: {
    label: 'PDF Report',
    icon: <PictureAsPdf />,
    description: 'Publication-ready PDF with formatted sections, charts, and appendix',
  },
  docx: {
    label: 'Word Document',
    icon: <Description />,
    description: 'Editable DOCX with TCFD-aligned section headings and styles',
  },
  xlsx: {
    label: 'Excel Workbook',
    icon: <TableChart />,
    description: 'Structured Excel with separate worksheets per pillar and data tables',
  },
  json: {
    label: 'JSON Data',
    icon: <Code />,
    description: 'Machine-readable JSON following TCFD disclosure schema',
  },
  xbrl: {
    label: 'XBRL Taxonomy',
    icon: <DataObject />,
    description: 'XBRL-tagged output for regulatory filing (ISSB/CSRD compatible)',
  },
};

const PILLAR_LABELS: Record<TCFDPillar, string> = {
  governance: 'Governance',
  strategy: 'Strategy',
  risk_management: 'Risk Management',
  metrics_targets: 'Metrics & Targets',
};

const DEFAULT_OPTIONS: ExportOptions = {
  format: 'pdf',
  pillars: ['governance', 'strategy', 'risk_management', 'metrics_targets'],
  includeCharts: true,
  includeEvidence: true,
  includeComplianceMatrix: true,
  includeAppendix: true,
  watermark: false,
  draft: false,
  reportTitle: 'TCFD Climate-Related Financial Disclosure',
  organizationName: '',
  reportingPeriod: '2024',
};

const ExportDialog: React.FC<ExportDialogProps> = ({
  open,
  onClose,
  onExport,
  isExporting = false,
  exportProgress = 0,
}) => {
  const [options, setOptions] = useState<ExportOptions>(DEFAULT_OPTIONS);

  const handleFormatChange = (format: ExportFormat) => {
    setOptions((prev) => ({ ...prev, format }));
  };

  const handlePillarToggle = (pillar: TCFDPillar) => {
    setOptions((prev) => ({
      ...prev,
      pillars: prev.pillars.includes(pillar)
        ? prev.pillars.filter((p) => p !== pillar)
        : [...prev.pillars, pillar],
    }));
  };

  const handleToggle = (field: keyof ExportOptions) => {
    setOptions((prev) => ({ ...prev, [field]: !prev[field] }));
  };

  const handleTextChange = (field: keyof ExportOptions, value: string) => {
    setOptions((prev) => ({ ...prev, [field]: value }));
  };

  const handleExport = () => {
    onExport(options.format, options);
  };

  const allPillarsSelected = options.pillars.length === 4;
  const noPillarsSelected = options.pillars.length === 0;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Download color="primary" />
          <Typography variant="h6">Export TCFD Disclosure</Typography>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {isExporting ? (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Generating {FORMAT_CONFIG[options.format].label}...
            </Typography>
            <LinearProgress
              variant="determinate"
              value={exportProgress}
              sx={{ height: 8, borderRadius: 4, mb: 2 }}
            />
            <Typography variant="body2" color="text.secondary">
              {exportProgress}% complete
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {/* Format Selection */}
            <Grid item xs={12}>
              <FormControl component="fieldset">
                <FormLabel sx={{ fontWeight: 600, mb: 1 }}>Export Format</FormLabel>
                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
                  {(Object.entries(FORMAT_CONFIG) as [ExportFormat, typeof FORMAT_CONFIG['pdf']][]).map(
                    ([format, config]) => (
                      <Box
                        key={format}
                        onClick={() => handleFormatChange(format)}
                        sx={{
                          flex: '1 0 150px',
                          maxWidth: 200,
                          p: 1.5,
                          border: options.format === format ? '2px solid' : '1px solid',
                          borderColor: options.format === format ? 'primary.main' : 'divider',
                          borderRadius: 1,
                          cursor: 'pointer',
                          backgroundColor: options.format === format ? 'primary.50' : 'transparent',
                          transition: 'all 0.2s',
                          '&:hover': { borderColor: 'primary.light', boxShadow: 1 },
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          {config.icon}
                          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            {config.label}
                          </Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.3 }}>
                          {config.description}
                        </Typography>
                      </Box>
                    )
                  )}
                </Box>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Divider />
            </Grid>

            {/* Report Metadata */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                size="small"
                label="Report Title"
                value={options.reportTitle}
                onChange={(e) => handleTextChange('reportTitle', e.target.value)}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Organization Name"
                value={options.organizationName}
                onChange={(e) => handleTextChange('organizationName', e.target.value)}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Reporting Period"
                value={options.reportingPeriod}
                onChange={(e) => handleTextChange('reportingPeriod', e.target.value)}
              />
            </Grid>

            <Grid item xs={12}>
              <Divider />
            </Grid>

            {/* Pillar Selection */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Include Pillars
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={allPillarsSelected}
                      indeterminate={!allPillarsSelected && !noPillarsSelected}
                      onChange={() => {
                        setOptions((prev) => ({
                          ...prev,
                          pillars: allPillarsSelected
                            ? []
                            : (['governance', 'strategy', 'risk_management', 'metrics_targets'] as TCFDPillar[]),
                        }));
                      }}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2" sx={{ fontWeight: 500 }}>Select All Pillars</Typography>}
                />
                {(Object.entries(PILLAR_LABELS) as [TCFDPillar, string][]).map(([pillar, label]) => (
                  <FormControlLabel
                    key={pillar}
                    control={
                      <Checkbox
                        checked={options.pillars.includes(pillar)}
                        onChange={() => handlePillarToggle(pillar)}
                        size="small"
                      />
                    }
                    label={<Typography variant="body2">{label}</Typography>}
                    sx={{ ml: 2 }}
                  />
                ))}
              </Box>
            </Grid>

            {/* Content Options */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Content Options
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.includeCharts}
                      onChange={() => handleToggle('includeCharts')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Include Charts & Visualizations</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.includeEvidence}
                      onChange={() => handleToggle('includeEvidence')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Include Supporting Evidence</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.includeComplianceMatrix}
                      onChange={() => handleToggle('includeComplianceMatrix')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Include Compliance Matrix</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.includeAppendix}
                      onChange={() => handleToggle('includeAppendix')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Include Data Appendix</Typography>}
                />

                <Divider sx={{ my: 1 }} />

                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.watermark}
                      onChange={() => handleToggle('watermark')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Add Confidential Watermark</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={options.draft}
                      onChange={() => handleToggle('draft')}
                      size="small"
                    />
                  }
                  label={<Typography variant="body2">Mark as Draft</Typography>}
                />
              </Box>
            </Grid>

            {/* Warnings */}
            {noPillarsSelected && (
              <Grid item xs={12}>
                <Alert severity="warning">
                  Please select at least one pillar to export.
                </Alert>
              </Grid>
            )}

            {options.format === 'xbrl' && (
              <Grid item xs={12}>
                <Alert severity="info">
                  XBRL export uses the IFRS Sustainability Disclosure Taxonomy (2024). Ensure all
                  required fields are populated for regulatory compliance.
                </Alert>
              </Grid>
            )}
          </Grid>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 'auto' }}>
          <Chip
            label={`${options.pillars.length} pillars selected`}
            size="small"
            variant="outlined"
          />
          <Chip
            label={FORMAT_CONFIG[options.format].label}
            size="small"
            color="primary"
          />
        </Box>
        <Button onClick={onClose} disabled={isExporting}>
          Cancel
        </Button>
        <Button
          variant="contained"
          startIcon={isExporting ? undefined : <Download />}
          onClick={handleExport}
          disabled={isExporting || noPillarsSelected}
        >
          {isExporting ? 'Exporting...' : 'Export'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ExportDialog;
