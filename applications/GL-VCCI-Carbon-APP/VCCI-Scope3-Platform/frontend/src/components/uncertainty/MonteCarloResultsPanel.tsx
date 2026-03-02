/**
 * MonteCarloResultsPanel - Comprehensive Monte Carlo results display
 *
 * Tabbed panel presenting MC simulation results across four views:
 * Summary (stat cards with mean, median, stdDev, CV, skewness, kurtosis),
 * Distribution (embedded histogram + KDE), Sensitivity (tornado diagram),
 * and Raw Data (scrollable sample table). Includes export controls for
 * PDF, CSV, and JSON, computation metadata, and data quality tier badges.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Paper,
  Typography,
  Box,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Stack,
  Divider,
  Alert,
} from '@mui/material';
import {
  Assessment,
  ShowChart,
  BarChart as BarChartIcon,
  TableChart,
  Download,
  Timer,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
} from '@mui/icons-material';
import type { MonteCarloResult, SensitivityParameter } from '../../store/slices/uncertaintySlice';
import UncertaintyDistribution from './UncertaintyDistribution';
import SensitivityTornado from './SensitivityTornado';
import { formatNumber } from '../../utils/formatters';

// ==============================================================================
// Types
// ==============================================================================

interface MonteCarloResultsPanelProps {
  result: MonteCarloResult;
  sensitivityData?: SensitivityParameter[];
}

interface TabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

// ==============================================================================
// Subcomponents
// ==============================================================================

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  if (value !== index) return null;
  return (
    <Box sx={{ py: 2 }} role="tabpanel">
      {children}
    </Box>
  );
};

interface StatItemProps {
  label: string;
  value: string;
  subtitle?: string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
}

const StatItem: React.FC<StatItemProps> = ({ label, value, subtitle, icon, color = 'primary' }) => (
  <Card variant="outlined">
    <CardContent sx={{ pb: '12px !important' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            {label}
          </Typography>
          <Typography variant="h5" component="div">
            {value}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="textSecondary">
              {subtitle}
            </Typography>
          )}
        </Box>
        <Box
          sx={{
            backgroundColor: (theme) => `${theme.palette[color].main}20`,
            borderRadius: 1.5,
            p: 1,
          }}
        >
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

// ==============================================================================
// Helpers
// ==============================================================================

const getTierConfig = (tier: number): { label: string; color: 'success' | 'warning' | 'error'; description: string } => {
  switch (tier) {
    case 1:
      return { label: 'Tier 1 - Primary Data', color: 'success', description: 'Supplier-specific or measured data' };
    case 2:
      return { label: 'Tier 2 - Secondary Data', color: 'warning', description: 'Industry-average or published EFs' };
    case 3:
      return { label: 'Tier 3 - Estimated Data', color: 'error', description: 'Spend-based or proxy estimates' };
    default:
      return { label: 'Unknown Tier', color: 'warning', description: 'Data quality not assessed' };
  }
};

const getSkewnessInterpretation = (skewness: number): string => {
  if (Math.abs(skewness) < 0.5) return 'Approximately symmetric';
  if (skewness > 0) return 'Right-skewed (positive tail)';
  return 'Left-skewed (negative tail)';
};

const getKurtosisInterpretation = (kurtosis: number): string => {
  if (Math.abs(kurtosis - 3) < 0.5) return 'Mesokurtic (normal-like)';
  if (kurtosis > 3) return 'Leptokurtic (heavy tails)';
  return 'Platykurtic (light tails)';
};

// ==============================================================================
// Component
// ==============================================================================

const MonteCarloResultsPanel: React.FC<MonteCarloResultsPanelProps> = ({
  result,
  sensitivityData = [],
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [rawDataPage, setRawDataPage] = useState(0);
  const [rawDataRowsPerPage, setRawDataRowsPerPage] = useState(25);

  const tierConfig = useMemo(() => getTierConfig(result.dataTier), [result.dataTier]);
  const cvPercent = useMemo(() => result.cv * 100, [result.cv]);

  // Paginated raw data samples
  const paginatedSamples = useMemo(() => {
    const displayData = result.distributionSamples.slice(0, 1000);
    const start = rawDataPage * rawDataRowsPerPage;
    return displayData.slice(start, start + rawDataRowsPerPage);
  }, [result.distributionSamples, rawDataPage, rawDataRowsPerPage]);

  // Export handlers
  const handleExportCSV = useCallback(() => {
    const header = 'sample_index,value\n';
    const rows = result.distributionSamples
      .map((val, idx) => `${idx + 1},${val}`)
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `mc_results_${result.id}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [result]);

  const handleExportJSON = useCallback(() => {
    const exportData = {
      id: result.id,
      statistics: {
        mean: result.mean,
        median: result.median,
        stdDev: result.stdDev,
        cv: result.cv,
        skewness: result.skewness,
        kurtosis: result.kurtosis,
      },
      percentiles: result.percentiles,
      metadata: {
        iterations: result.iterations,
        convergenceAchieved: result.convergenceAchieved,
        computationTimeMs: result.computationTimeMs,
        dataTier: result.dataTier,
        unit: result.unit,
      },
      samples: result.distributionSamples,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `mc_results_${result.id}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header with metadata */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            Monte Carlo Simulation Results
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Chip
              icon={result.convergenceAchieved ? <CheckCircle /> : <Warning />}
              label={result.convergenceAchieved ? 'Converged' : 'Not Converged'}
              size="small"
              color={result.convergenceAchieved ? 'success' : 'warning'}
            />
            <Chip
              icon={<Timer />}
              label={`${(result.computationTimeMs / 1000).toFixed(1)}s`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`${formatNumber(result.iterations)} iterations`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={tierConfig.label}
              size="small"
              color={tierConfig.color}
              variant="outlined"
            />
          </Stack>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<Download />}
            onClick={handleExportCSV}
          >
            CSV
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<Download />}
            onClick={handleExportJSON}
          >
            JSON
          </Button>
        </Stack>
      </Box>

      {/* CV Warning */}
      {cvPercent > 50 && (
        <Alert severity="warning" sx={{ mb: 2 }} icon={<ErrorIcon />}>
          High coefficient of variation ({cvPercent.toFixed(1)}%). Results have significant
          uncertainty. Consider collecting higher-quality data (Tier 1) to reduce variability.
        </Alert>
      )}

      <Divider />

      {/* Tabs */}
      <Tabs
        value={activeTab}
        onChange={(_, val) => setActiveTab(val)}
        sx={{ borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab icon={<Assessment />} label="Summary" iconPosition="start" />
        <Tab icon={<ShowChart />} label="Distribution" iconPosition="start" />
        <Tab icon={<BarChartIcon />} label="Sensitivity" iconPosition="start" />
        <Tab icon={<TableChart />} label="Raw Data" iconPosition="start" />
      </Tabs>

      {/* Summary Tab */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Mean"
              value={`${formatNumber(result.mean, 2)} ${result.unit}`}
              subtitle="Expected value"
              icon={<Assessment sx={{ color: 'primary.main' }} />}
              color="primary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Median (P50)"
              value={`${formatNumber(result.median, 2)} ${result.unit}`}
              subtitle="50th percentile"
              icon={<ShowChart sx={{ color: 'secondary.main' }} />}
              color="secondary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Standard Deviation"
              value={`${formatNumber(result.stdDev, 2)} ${result.unit}`}
              subtitle={`+/- 1 sigma range`}
              icon={<BarChartIcon sx={{ color: 'info.main' }} />}
              color="info"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Coefficient of Variation"
              value={`${cvPercent.toFixed(1)}%`}
              subtitle={cvPercent > 50 ? 'High variability' : cvPercent > 25 ? 'Moderate variability' : 'Low variability'}
              icon={cvPercent > 50 ? <Warning sx={{ color: 'warning.main' }} /> : <CheckCircle sx={{ color: 'success.main' }} />}
              color={cvPercent > 50 ? 'warning' : cvPercent > 25 ? 'info' : 'success'}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Skewness"
              value={result.skewness.toFixed(4)}
              subtitle={getSkewnessInterpretation(result.skewness)}
              icon={<ShowChart sx={{ color: 'primary.main' }} />}
              color="primary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <StatItem
              label="Kurtosis"
              value={result.kurtosis.toFixed(4)}
              subtitle={getKurtosisInterpretation(result.kurtosis)}
              icon={<Assessment sx={{ color: 'secondary.main' }} />}
              color="secondary"
            />
          </Grid>
        </Grid>

        {/* Percentile breakdown */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Percentile Breakdown
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Percentile</TableCell>
                  <TableCell align="right">Value ({result.unit})</TableCell>
                  <TableCell align="right">Deviation from Mean</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {[
                  { label: 'P5', value: result.percentiles.p5 },
                  { label: 'P10', value: result.percentiles.p10 },
                  { label: 'P25', value: result.percentiles.p25 },
                  { label: 'P50 (Median)', value: result.percentiles.p50 },
                  { label: 'P75', value: result.percentiles.p75 },
                  { label: 'P90', value: result.percentiles.p90 },
                  { label: 'P95', value: result.percentiles.p95 },
                ].map((row) => (
                  <TableRow key={row.label}>
                    <TableCell>{row.label}</TableCell>
                    <TableCell align="right">{formatNumber(row.value, 2)}</TableCell>
                    <TableCell align="right">
                      {((row.value - result.mean) / result.mean * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>

        {/* Computation metadata */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Computation Details
          </Typography>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            <Chip label={`Iterations: ${formatNumber(result.iterations)}`} size="small" variant="outlined" />
            <Chip label={`Time: ${(result.computationTimeMs / 1000).toFixed(2)}s`} size="small" variant="outlined" />
            <Chip
              label={`Convergence: ${result.convergenceAchieved ? 'Yes' : 'No'} (threshold: ${result.convergenceThreshold})`}
              size="small"
              variant="outlined"
              color={result.convergenceAchieved ? 'success' : 'warning'}
            />
            <Chip label={tierConfig.description} size="small" color={tierConfig.color} variant="outlined" />
          </Stack>
        </Box>
      </TabPanel>

      {/* Distribution Tab */}
      <TabPanel value={activeTab} index={1}>
        <UncertaintyDistribution
          data={result.distributionSamples}
          mean={result.mean}
          median={result.median}
          p5={result.percentiles.p5}
          p50={result.percentiles.p50}
          p95={result.percentiles.p95}
          stdDev={result.stdDev}
          title="Simulation Distribution"
          unit={result.unit}
        />
      </TabPanel>

      {/* Sensitivity Tab */}
      <TabPanel value={activeTab} index={2}>
        {sensitivityData.length > 0 ? (
          <SensitivityTornado
            parameters={sensitivityData}
            baseline={result.mean}
            title="Parameter Sensitivity"
          />
        ) : (
          <Alert severity="info">
            No sensitivity analysis data available. Run a sensitivity analysis to see
            which parameters have the greatest impact on results.
          </Alert>
        )}
      </TabPanel>

      {/* Raw Data Tab */}
      <TabPanel value={activeTab} index={3}>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
          Showing first {Math.min(1000, result.distributionSamples.length).toLocaleString()} of{' '}
          {result.distributionSamples.length.toLocaleString()} samples
        </Typography>
        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Sample #</TableCell>
                <TableCell align="right">Value ({result.unit})</TableCell>
                <TableCell align="right">Z-Score</TableCell>
                <TableCell align="center">Percentile Region</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginatedSamples.map((value, idx) => {
                const sampleIndex = rawDataPage * rawDataRowsPerPage + idx + 1;
                const zScore = (value - result.mean) / result.stdDev;
                let region = 'P25-P75';
                if (value <= result.percentiles.p5) region = 'Below P5';
                else if (value <= result.percentiles.p25) region = 'P5-P25';
                else if (value >= result.percentiles.p95) region = 'Above P95';
                else if (value >= result.percentiles.p75) region = 'P75-P95';

                return (
                  <TableRow key={sampleIndex} hover>
                    <TableCell>{sampleIndex}</TableCell>
                    <TableCell align="right">{value.toFixed(4)}</TableCell>
                    <TableCell align="right">{zScore.toFixed(3)}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={region}
                        size="small"
                        color={
                          region === 'P25-P75' ? 'success' :
                          region === 'P5-P25' || region === 'P75-P95' ? 'warning' :
                          'error'
                        }
                        variant="outlined"
                      />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={Math.min(1000, result.distributionSamples.length)}
          page={rawDataPage}
          onPageChange={(_, page) => setRawDataPage(page)}
          rowsPerPage={rawDataRowsPerPage}
          onRowsPerPageChange={(e) => {
            setRawDataRowsPerPage(parseInt(e.target.value, 10));
            setRawDataPage(0);
          }}
          rowsPerPageOptions={[10, 25, 50, 100]}
        />
      </TabPanel>
    </Paper>
  );
};

export default MonteCarloResultsPanel;
