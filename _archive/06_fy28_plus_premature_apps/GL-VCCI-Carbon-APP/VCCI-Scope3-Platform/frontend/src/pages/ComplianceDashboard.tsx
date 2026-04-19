/**
 * ComplianceDashboard - Multi-standard compliance overview page
 *
 * Displays the ComplianceScorecard, a radar chart of coverage across
 * 5 standards, a compliance trend line chart, and an action items table
 * with gap severity and status tracking.
 */

import React, { useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
} from '@mui/material';
import { Download, Refresh, CheckCircle, Schedule, HourglassEmpty } from '@mui/icons-material';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from 'recharts';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchComplianceScorecard,
  fetchComplianceGaps,
  updateActionItemStatus,
} from '../store/slices/complianceSlice';
import { ComplianceScorecard } from '../components/cdp';
import LoadingSpinner from '../components/LoadingSpinner';

// =============================================================================
// Severity Configuration
// =============================================================================

const SEVERITY_COLORS: Record<string, 'error' | 'warning' | 'info' | 'default'> = {
  critical: 'error',
  high: 'warning',
  medium: 'info',
  low: 'default',
};

const STATUS_ICONS: Record<string, React.ReactElement> = {
  open: <HourglassEmpty fontSize="small" color="error" />,
  in_progress: <Schedule fontSize="small" color="warning" />,
  resolved: <CheckCircle fontSize="small" color="success" />,
};

// =============================================================================
// Mock trend data (in production would come from the API)
// =============================================================================

const COMPLIANCE_TREND_DATA = [
  { month: 'Jul', 'GHG Protocol': 45, 'ESRS E1': 35, CDP: 40, 'IFRS S2': 30, 'ISO 14083': 25 },
  { month: 'Aug', 'GHG Protocol': 52, 'ESRS E1': 42, CDP: 45, 'IFRS S2': 38, 'ISO 14083': 32 },
  { month: 'Sep', 'GHG Protocol': 58, 'ESRS E1': 50, CDP: 52, 'IFRS S2': 44, 'ISO 14083': 40 },
  { month: 'Oct', 'GHG Protocol': 65, 'ESRS E1': 58, CDP: 60, 'IFRS S2': 52, 'ISO 14083': 48 },
  { month: 'Nov', 'GHG Protocol': 72, 'ESRS E1': 65, CDP: 68, 'IFRS S2': 60, 'ISO 14083': 55 },
  { month: 'Dec', 'GHG Protocol': 78, 'ESRS E1': 72, CDP: 74, 'IFRS S2': 68, 'ISO 14083': 62 },
  { month: 'Jan', 'GHG Protocol': 82, 'ESRS E1': 76, CDP: 78, 'IFRS S2': 72, 'ISO 14083': 68 },
  { month: 'Feb', 'GHG Protocol': 85, 'ESRS E1': 80, CDP: 82, 'IFRS S2': 76, 'ISO 14083': 72 },
];

const TREND_LINE_COLORS = ['#1976d2', '#2e7d32', '#ed6c02', '#9c27b0', '#00838f'];

// =============================================================================
// Main Component
// =============================================================================

const ComplianceDashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { scorecard, actionItems, loading, error } = useAppSelector(
    (state) => state.compliance
  );

  // Load data on mount
  useEffect(() => {
    dispatch(fetchComplianceScorecard());
    dispatch(fetchComplianceGaps());
  }, [dispatch]);

  // Radar chart data
  const radarData = useMemo(() => {
    if (!scorecard) return [];
    return scorecard.standards.map((s) => ({
      standard: s.shortName,
      coverage: s.coveragePercentage,
      fullMark: 100,
    }));
  }, [scorecard]);

  // Handlers
  const handleRefresh = useCallback(() => {
    dispatch(fetchComplianceScorecard());
    dispatch(fetchComplianceGaps());
  }, [dispatch]);

  const handleExportReport = useCallback(() => {
    // In production this would trigger a report download
    window.alert('Compliance report export will be implemented with the backend API.');
  }, []);

  const handleStatusToggle = useCallback(
    (id: string, currentStatus: string) => {
      const nextStatus =
        currentStatus === 'open'
          ? 'in_progress'
          : currentStatus === 'in_progress'
          ? 'resolved'
          : 'open';
      dispatch(
        updateActionItemStatus({
          id,
          status: nextStatus as 'open' | 'in_progress' | 'resolved',
        })
      );
    },
    [dispatch]
  );

  // Loading state
  if (loading && !scorecard) {
    return <LoadingSpinner message="Loading compliance data..." />;
  }

  return (
    <Box>
      {/* Page Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h4">Compliance Dashboard</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Download />}
            onClick={handleExportReport}
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Error state */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Compliance Scorecard */}
      {scorecard && (
        <Box sx={{ mb: 3 }}>
          <ComplianceScorecard
            standards={scorecard.standards}
            overallScore={scorecard.overallScore}
            onExportReport={handleExportReport}
          />
        </Box>
      )}

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Radar Chart */}
        <Grid item xs={12} md={5}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Coverage by Standard
              </Typography>
              {radarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={320}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                    <PolarGrid stroke="#e0e0e0" />
                    <PolarAngleAxis
                      dataKey="standard"
                      tick={{ fontSize: 12, fill: '#666' }}
                    />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 100]}
                      tick={{ fontSize: 10 }}
                    />
                    <Radar
                      name="Coverage %"
                      dataKey="coverage"
                      stroke="#1976d2"
                      fill="#1976d2"
                      fillOpacity={0.25}
                      strokeWidth={2}
                    />
                    <RechartsTooltip
                      formatter={(value: number) => [`${value.toFixed(1)}%`, 'Coverage']}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <Box
                  sx={{
                    height: 320,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography color="text.secondary">No data available</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Compliance Trend Chart */}
        <Grid item xs={12} md={7}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Compliance Trend (Last 8 Months)
              </Typography>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={COMPLIANCE_TREND_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 12 }}
                    label={{
                      value: 'Coverage %',
                      angle: -90,
                      position: 'insideLeft',
                      style: { fontSize: 12 },
                    }}
                  />
                  <RechartsTooltip formatter={(value: number) => [`${value}%`]} />
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    iconType="line"
                    wrapperStyle={{ fontSize: '0.75rem' }}
                  />
                  {['GHG Protocol', 'ESRS E1', 'CDP', 'IFRS S2', 'ISO 14083'].map(
                    (key, index) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={TREND_LINE_COLORS[index]}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                    )
                  )}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Action Items Table */}
      <Card variant="outlined">
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 2,
            }}
          >
            <Typography variant="h6">Action Items</Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label={`Open: ${actionItems.filter((i) => i.status === 'open').length}`}
                color="error"
                size="small"
                variant="outlined"
              />
              <Chip
                label={`In Progress: ${actionItems.filter((i) => i.status === 'in_progress').length}`}
                color="warning"
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Resolved: ${actionItems.filter((i) => i.status === 'resolved').length}`}
                color="success"
                size="small"
                variant="outlined"
              />
            </Box>
          </Box>

          {actionItems.length === 0 ? (
            <Alert severity="success">
              No compliance gaps found. All requirements are currently met.
            </Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Gap</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Standard</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', width: 100 }}>Severity</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Action Required</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', width: 120 }}>Status</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', width: 110 }}>Due Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {actionItems.map((item) => (
                    <TableRow
                      key={item.id}
                      hover
                      sx={{
                        backgroundColor:
                          item.status === 'resolved'
                            ? 'rgba(46, 125, 50, 0.04)'
                            : item.severity === 'critical'
                            ? 'rgba(211, 47, 47, 0.04)'
                            : 'inherit',
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }} title={item.gap}>
                          {item.gap}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={item.standard}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem' }}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={item.severity}
                          size="small"
                          color={SEVERITY_COLORS[item.severity] || 'default'}
                          sx={{ fontSize: '0.7rem', textTransform: 'capitalize' }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          noWrap
                          sx={{ maxWidth: 250 }}
                          title={item.action}
                        >
                          {item.action}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Tooltip title="Click to change status">
                          <Chip
                            icon={STATUS_ICONS[item.status]}
                            label={item.status.replace('_', ' ')}
                            size="small"
                            variant="outlined"
                            onClick={() => handleStatusToggle(item.id, item.status)}
                            sx={{
                              fontSize: '0.7rem',
                              textTransform: 'capitalize',
                              cursor: 'pointer',
                            }}
                          />
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {item.dueDate
                            ? new Date(item.dueDate).toLocaleDateString()
                            : '--'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ComplianceDashboard;
