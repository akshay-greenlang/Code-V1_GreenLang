/**
 * RiskAssessment - Page for viewing risk matrices, alerts, and evidence.
 *
 * Displays the RiskMatrix (country x commodity), risk alert feed filtered
 * by severity, and a side panel with RiskTimeline + EvidenceViewer for
 * the selected plot or supplier.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Drawer,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Button,
  SelectChangeEvent,
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RiskMatrix from '../components/risk/RiskMatrix';
import RiskTimeline from '../components/risk/RiskTimeline';
import EvidenceViewer, { SatelliteEvidence } from '../components/risk/EvidenceViewer';
import apiClient from '../services/api';
import type { RiskHeatmapEntry, RiskAlert, RiskTrendPoint, RiskLevel } from '../types';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const RISK_COLORS: Record<RiskLevel, string> = {
  low: '#4caf50',
  standard: '#2196f3',
  high: '#ff9800',
  critical: '#f44336',
};

const RiskAssessmentPage: React.FC = () => {
  // Heatmap
  const [heatmapData, setHeatmapData] = useState<Record<string, Record<string, number>>>({});
  const [countryNames, setCountryNames] = useState<Record<string, string>>({});

  // Alerts
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [alertFilter, setAlertFilter] = useState<string>('');

  // Detail panel
  const [selectedSupplierId, setSelectedSupplierId] = useState<string | null>(null);
  const [selectedPlotName, setSelectedPlotName] = useState<string>('');
  const [trendData, setTrendData] = useState<RiskTrendPoint[]>([]);
  const [evidence, setEvidence] = useState<SatelliteEvidence | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch data
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [heatmapEntries, alertsRes] = await Promise.all([
          apiClient.getRiskHeatmap(),
          apiClient.getRiskAlerts({ per_page: 50 }),
        ]);

        // Transform heatmap entries into Record<country, Record<commodity, score>>
        const hm: Record<string, Record<string, number>> = {};
        const cn: Record<string, string> = {};
        heatmapEntries.forEach((entry: RiskHeatmapEntry) => {
          if (!hm[entry.country]) hm[entry.country] = {};
          hm[entry.country][entry.commodity] = entry.risk_score;
          cn[entry.country] = entry.country;
        });
        setHeatmapData(hm);
        setCountryNames(cn);
        setAlerts(alertsRes.items);
      } catch {
        setError('Failed to load risk data.');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Load trends when supplier selected
  useEffect(() => {
    if (!selectedSupplierId) return;
    apiClient
      .getRiskTrends(selectedSupplierId, '12m')
      .then((data) => setTrendData(data))
      .catch(() => setTrendData([]));
  }, [selectedSupplierId]);

  // Handle cell click on matrix
  const handleCellClick = (country: string, commodity: string, score: number) => {
    // Find an alert for this country/commodity for demonstration
    const matchingAlert = alerts.find(
      (a) => a.supplier_name.toLowerCase().includes(country.toLowerCase())
    );
    if (matchingAlert) {
      setSelectedSupplierId(matchingAlert.supplier_id);
      setSelectedPlotName(`${country} - ${commodity}`);
    }
  };

  // Handle alert click
  const handleAlertClick = (alert: RiskAlert) => {
    setSelectedSupplierId(alert.supplier_id);
    setSelectedPlotName(alert.supplier_name);

    // Mock evidence for demonstration
    setEvidence({
      plot_id: alert.id,
      plot_name: alert.supplier_name,
      baseline_date: '2020-01-01',
      assessment_date: alert.created_at,
      ndvi_baseline: 0.72,
      ndvi_current: alert.severity === 'critical' ? 0.35 : 0.58,
      ndvi_change: alert.severity === 'critical' ? -0.37 : -0.14,
      forest_cover_baseline_pct: 88.2,
      forest_cover_current_pct: alert.severity === 'critical' ? 52.1 : 76.8,
      tree_loss_pct: alert.severity === 'critical' ? 36.1 : 11.4,
      confidence_score: 0.89,
      classification:
        alert.severity === 'critical'
          ? 'deforestation'
          : alert.severity === 'high'
          ? 'degradation'
          : 'no_change',
      satellite_source: 'Sentinel-2',
      resolution_m: 10,
    });
  };

  // Filtered alerts
  const filteredAlerts = alertFilter
    ? alerts.filter((a) => a.severity === alertFilter)
    : alerts;

  const handleResolveAlert = async (alertId: string) => {
    try {
      await apiClient.resolveRiskAlert(alertId);
      setAlerts((prev) => prev.filter((a) => a.id !== alertId));
    } catch {
      // Silently fail
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', pt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Typography variant="h4" fontWeight={700} mb={3}>
        Risk Assessment
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Risk Matrix */}
      <Box mb={3}>
        <RiskMatrix
          heatmapData={heatmapData}
          countryNames={countryNames}
          onCellClick={handleCellClick}
        />
      </Box>

      {/* Alerts feed */}
      <Card>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
            <Typography variant="h6">Risk Alerts</Typography>
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <InputLabel>Severity</InputLabel>
              <Select
                value={alertFilter}
                label="Severity"
                onChange={(e: SelectChangeEvent) => setAlertFilter(e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="standard">Standard</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>
          </Stack>
          <Divider sx={{ mb: 1 }} />
          {filteredAlerts.length === 0 ? (
            <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
              No risk alerts.
            </Typography>
          ) : (
            <List dense disablePadding>
              {filteredAlerts.slice(0, 15).map((alert) => (
                <ListItem
                  key={alert.id}
                  divider
                  button
                  onClick={() => handleAlertClick(alert)}
                  secondaryAction={
                    !alert.is_resolved && (
                      <Button
                        size="small"
                        onClick={(e) => { e.stopPropagation(); handleResolveAlert(alert.id); }}
                      >
                        Resolve
                      </Button>
                    )
                  }
                >
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    {alert.severity === 'critical' ? (
                      <ErrorIcon color="error" fontSize="small" />
                    ) : alert.severity === 'high' ? (
                      <WarningAmberIcon color="warning" fontSize="small" />
                    ) : (
                      <InfoIcon color="info" fontSize="small" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="body2" fontWeight={500}>{alert.title}</Typography>
                        <Chip
                          label={alert.severity}
                          size="small"
                          sx={{
                            backgroundColor: RISK_COLORS[alert.severity],
                            color: '#fff',
                            textTransform: 'capitalize',
                            fontWeight: 600,
                            height: 20,
                            fontSize: 10,
                          }}
                        />
                      </Stack>
                    }
                    secondary={`${alert.supplier_name} | ${new Date(alert.created_at).toLocaleDateString('en-GB')}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      {/* Detail Drawer */}
      <Drawer
        anchor="right"
        open={Boolean(selectedSupplierId)}
        onClose={() => {
          setSelectedSupplierId(null);
          setTrendData([]);
          setEvidence(null);
        }}
        PaperProps={{ sx: { width: { xs: '100%', md: 560 }, p: 2 } }}
      >
        <Typography variant="h6" gutterBottom>{selectedPlotName}</Typography>
        <Divider sx={{ mb: 2 }} />

        {/* Risk Timeline */}
        <Box mb={2}>
          <RiskTimeline trends={trendData} title="Risk Score Over Time" />
        </Box>

        {/* Evidence Viewer */}
        {evidence && (
          <EvidenceViewer evidence={evidence} />
        )}

        {!evidence && trendData.length === 0 && (
          <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
            Select an alert or matrix cell to view detailed risk analysis.
          </Typography>
        )}
      </Drawer>
    </Box>
  );
};

export default RiskAssessmentPage;
