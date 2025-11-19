/**
 * Operations Dashboard Component
 *
 * Real-time furnace operations monitoring with 20+ KPIs
 * Live temperature, fuel consumption, alerts, and control recommendations
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert as MuiAlert,
  CircularProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  IconButton,
  Badge,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Thermostat,
  LocalFireDepartment,
  Speed,
  Opacity,
  ElectricBolt,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Notifications,
  TrendingUp,
  TrendingDown,
  Refresh,
  WaterDrop,
  Air,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';

import KPICard from '../charts/KPICard';
import GaugeChart from '../charts/GaugeChart';
import { useFurnaceStore } from '../../store/furnaceStore';
import { apiClient } from '../../services/apiClient';
import { useWebSocket } from '../../services/websocket';
import type {
  FurnacePerformance,
  Alert,
  ZonePerformance,
  SensorReading,
} from '../../types';

const OperationsDashboard: React.FC = () => {
  const { selectedFurnaceId, activeAlerts } = useFurnaceStore();
  const [performance, setPerformance] = useState<FurnacePerformance | null>(null);
  const [zoneTemperatures, setZoneTemperatures] = useState<Map<string, number[]>>(
    new Map()
  );

  // Fetch initial performance data
  const {
    data: initialPerformance,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['performance', selectedFurnaceId],
    queryFn: () =>
      selectedFurnaceId ? apiClient.getPerformance(selectedFurnaceId) : null,
    enabled: !!selectedFurnaceId,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Real-time updates via WebSocket
  const { service, isConnected } = useWebSocket({
    furnaceId: selectedFurnaceId || '',
    autoConnect: !!selectedFurnaceId,
  });

  useEffect(() => {
    if (!selectedFurnaceId) return;

    // Performance updates
    const unsubscribePerformance = service.on(
      'performance_update',
      (data: FurnacePerformance) => {
        setPerformance(data);
      }
    );

    // Sensor readings for temperature trends
    const unsubscribeSensor = service.on('sensor_reading', (reading: SensorReading) => {
      if (reading.sensorId.includes('temperature')) {
        setZoneTemperatures((prev) => {
          const newMap = new Map(prev);
          const history = newMap.get(reading.sensorId) || [];
          newMap.set(reading.sensorId, [...history, reading.value].slice(-20));
          return newMap;
        });
      }
    });

    return () => {
      unsubscribePerformance();
      unsubscribeSensor();
    };
  }, [selectedFurnaceId, service]);

  // Use real-time data if available, otherwise use initial data
  const currentPerformance = performance || initialPerformance;

  if (!selectedFurnaceId) {
    return (
      <Box sx={{ p: 3 }}>
        <MuiAlert severity="info">
          Please select a furnace to view operations data.
        </MuiAlert>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <MuiAlert severity="error">
          Failed to load performance data: {error.message}
        </MuiAlert>
      </Box>
    );
  }

  if (!currentPerformance) {
    return (
      <Box sx={{ p: 3 }}>
        <MuiAlert severity="warning">No performance data available.</MuiAlert>
      </Box>
    );
  }

  const kpis = currentPerformance.kpis;
  const thermal = currentPerformance.thermal;
  const fuel = currentPerformance.fuel;
  const production = currentPerformance.production;
  const emissions = currentPerformance.emissions;
  const efficiency = currentPerformance.efficiency;

  // Get active alerts for this furnace
  const furnaceAlerts = activeAlerts.filter((a) => a.furnaceId === selectedFurnaceId);
  const criticalAlerts = furnaceAlerts.filter((a) => a.severity === 'critical');

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h4" gutterBottom>
            Operations Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Last updated: {new Date(currentPerformance.timestamp).toLocaleString()}
            </Typography>
            <Chip
              icon={isConnected ? <CheckCircle /> : <ErrorIcon />}
              label={isConnected ? 'Live' : 'Disconnected'}
              color={isConnected ? 'success' : 'error'}
              size="small"
            />
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton onClick={() => refetch()}>
            <Refresh />
          </IconButton>
          <Badge badgeContent={criticalAlerts.length} color="error">
            <IconButton>
              <Notifications />
            </IconButton>
          </Badge>
        </Box>
      </Box>

      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <MuiAlert severity="error" sx={{ mb: 3 }}>
          <strong>{criticalAlerts.length} Critical Alert(s):</strong>{' '}
          {criticalAlerts.map((a) => a.title).join(', ')}
        </MuiAlert>
      )}

      {/* Primary KPIs Grid - 6 columns */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Overall Efficiency"
            value={kpis.overallEfficiency}
            unit="%"
            target={95}
            icon={<Speed />}
            status={kpis.overallEfficiency >= 90 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Production Rate"
            value={kpis.productionRate}
            unit="t/hr"
            target={production.target}
            icon={<TrendingUp />}
            status={production.achievement >= 90 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Avg Temperature"
            value={thermal.averageTemperature}
            unit="°C"
            icon={<Thermostat />}
            status="good"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Fuel Flow"
            value={fuel.flowRate}
            unit="kg/hr"
            icon={<LocalFireDepartment />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="SEC"
            value={kpis.specificEnergyConsumption}
            unit="MJ/t"
            icon={<ElectricBolt />}
            trend="decreasing"
            trendValue={-2.1}
            status="good"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Carbon Intensity"
            value={kpis.carbonIntensity}
            unit="kgCO₂/t"
            icon={<Air />}
            trend="decreasing"
            trendValue={-3.5}
            status="good"
          />
        </Grid>
      </Grid>

      {/* Secondary KPIs Grid - Efficiency Metrics */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Thermal Efficiency"
            value={kpis.thermalEfficiency}
            unit="%"
            target={efficiency.thermal.target}
            status={kpis.thermalEfficiency >= 85 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Fuel Efficiency"
            value={kpis.fuelEfficiency}
            unit="%"
            target={90}
            status={kpis.fuelEfficiency >= 85 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Availability"
            value={kpis.availabilityFactor}
            unit="%"
            target={95}
            status={kpis.availabilityFactor >= 95 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Utilization"
            value={kpis.utilizationRate}
            unit="%"
            target={90}
            status={kpis.utilizationRate >= 85 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="Quality Index"
            value={kpis.qualityIndex}
            unit="%"
            target={98}
            status={kpis.qualityIndex >= 95 ? 'good' : 'warning'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <KPICard
            title="OEE"
            value={efficiency.overall.oee}
            unit="%"
            target={85}
            status={efficiency.overall.oee >= 80 ? 'good' : 'warning'}
          />
        </Grid>
      </Grid>

      {/* Temperature & Process Control Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Zone Temperature Monitoring */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Multi-Zone Temperature Monitoring
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line
                  data={{
                    labels: Array.from({ length: 20 }, (_, i) => i + 1),
                    datasets: thermal.zones.map((zone, idx) => ({
                      label: zone.zoneId,
                      data:
                        zoneTemperatures.get(`zone-${idx}-temp`) ||
                        Array.from({ length: 20 }, () => zone.currentTemperature),
                      borderColor: `hsl(${(idx * 360) / thermal.zones.length}, 70%, 50%)`,
                      backgroundColor: `hsla(${
                        (idx * 360) / thermal.zones.length
                      }, 70%, 50%, 0.2)`,
                      tension: 0.4,
                    })),
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { position: 'top' as const },
                      tooltip: {
                        mode: 'index',
                        intersect: false,
                      },
                    },
                    scales: {
                      y: {
                        title: {
                          display: true,
                          text: 'Temperature (°C)',
                        },
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Temperature Uniformity & Hot Spots */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Temperature Control
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Temperature Uniformity
                </Typography>
                <GaugeChart
                  value={thermal.temperatureUniformity}
                  maxValue={100}
                  unit="%"
                  thresholds={{ good: 95, warning: 90, critical: 85 }}
                  size={150}
                />
              </Box>
              <Divider sx={{ my: 2 }} />
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Hot Spots Detected
                </Typography>
                <Typography variant="h4" color="error.main">
                  {thermal.hotSpots.length}
                </Typography>
                {thermal.hotSpots.slice(0, 2).map((spot, idx) => (
                  <Chip
                    key={idx}
                    label={`${spot.location}: ${spot.temperature}°C`}
                    size="small"
                    color="error"
                    sx={{ mt: 1, mr: 1 }}
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Zone Performance Table */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Zone-by-Zone Performance
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Zone</TableCell>
                      <TableCell align="right">Current (°C)</TableCell>
                      <TableCell align="right">Target (°C)</TableCell>
                      <TableCell align="right">Deviation</TableCell>
                      <TableCell align="right">Stability</TableCell>
                      <TableCell align="right">Control Perf.</TableCell>
                      <TableCell align="center">Trend</TableCell>
                      <TableCell align="center">Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {thermal.zones.map((zone: ZonePerformance) => (
                      <TableRow key={zone.zoneId}>
                        <TableCell>{zone.zoneId}</TableCell>
                        <TableCell align="right">
                          {zone.currentTemperature.toFixed(1)}
                        </TableCell>
                        <TableCell align="right">
                          {zone.targetTemperature.toFixed(1)}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            color:
                              Math.abs(zone.deviation) > 10
                                ? 'error.main'
                                : Math.abs(zone.deviation) > 5
                                ? 'warning.main'
                                : 'success.main',
                          }}
                        >
                          {zone.deviation > 0 ? '+' : ''}
                          {zone.deviation.toFixed(1)}
                        </TableCell>
                        <TableCell align="right">{zone.stability.toFixed(1)}%</TableCell>
                        <TableCell align="right">
                          {zone.controlPerformance.toFixed(1)}%
                        </TableCell>
                        <TableCell align="center">
                          {zone.trend === 'increasing' && (
                            <TrendingUp color="error" fontSize="small" />
                          )}
                          {zone.trend === 'decreasing' && (
                            <TrendingDown color="primary" fontSize="small" />
                          )}
                          {zone.trend === 'stable' && (
                            <CheckCircle color="success" fontSize="small" />
                          )}
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={
                              Math.abs(zone.deviation) <= 5 &&
                              zone.stability >= 95
                                ? 'Good'
                                : Math.abs(zone.deviation) <= 10
                                ? 'Warning'
                                : 'Critical'
                            }
                            size="small"
                            color={
                              Math.abs(zone.deviation) <= 5 &&
                              zone.stability >= 95
                                ? 'success'
                                : Math.abs(zone.deviation) <= 10
                                ? 'warning'
                                : 'error'
                            }
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Fuel & Emissions Monitoring */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Fuel Consumption
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Current Rate
                  </Typography>
                  <Typography variant="h5">
                    {fuel.consumption.current.toFixed(0)}
                  </Typography>
                  <Typography variant="caption">kg/hr</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Specific
                  </Typography>
                  <Typography variant="h5">
                    {fuel.consumption.specific.toFixed(1)}
                  </Typography>
                  <Typography variant="caption">kg/tonne</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Pressure
                  </Typography>
                  <Typography variant="h5">{fuel.pressure.toFixed(1)}</Typography>
                  <Typography variant="caption">bar</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Temperature
                  </Typography>
                  <Typography variant="h5">{fuel.temperature.toFixed(0)}</Typography>
                  <Typography variant="caption">°C</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Combustion Efficiency
              </Typography>
              <GaugeChart
                value={efficiency.combustion.efficiency}
                maxValue={100}
                title="Combustion Eff."
                unit="%"
                thresholds={{ good: 95, warning: 90, critical: 85 }}
                size={150}
              />
              <Grid container spacing={2} sx={{ mt: 2 }}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Excess Air
                  </Typography>
                  <Typography variant="h6">
                    {efficiency.combustion.excessAir.toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    O₂ Level
                  </Typography>
                  <Typography variant="h6">
                    {efficiency.combustion.o2Level.toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Emissions Status
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    CO₂
                  </Typography>
                  <Typography variant="h6">{emissions.co2.current.toFixed(0)}</Typography>
                  <Chip
                    label={`${emissions.co2.compliance.toFixed(0)}%`}
                    size="small"
                    color={emissions.co2.compliance >= 95 ? 'success' : 'error'}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    NOx
                  </Typography>
                  <Typography variant="h6">{emissions.nox.current.toFixed(1)}</Typography>
                  <Chip
                    label={`${emissions.nox.compliance.toFixed(0)}%`}
                    size="small"
                    color={emissions.nox.compliance >= 95 ? 'success' : 'error'}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    SOx
                  </Typography>
                  <Typography variant="h6">{emissions.sox.current.toFixed(1)}</Typography>
                  <Chip
                    label={`${emissions.sox.compliance.toFixed(0)}%`}
                    size="small"
                    color={emissions.sox.compliance >= 95 ? 'success' : 'error'}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Particulates
                  </Typography>
                  <Typography variant="h6">
                    {emissions.particulates.current.toFixed(1)}
                  </Typography>
                  <Chip
                    label={`${emissions.particulates.compliance.toFixed(0)}%`}
                    size="small"
                    color={emissions.particulates.compliance >= 95 ? 'success' : 'error'}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Live Alert Feed */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Production Metrics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Hourly Rate
                  </Typography>
                  <Typography variant="h5">
                    {production.throughput.hourly.toFixed(1)}
                  </Typography>
                  <Typography variant="caption">tonnes/hr</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Daily Output
                  </Typography>
                  <Typography variant="h5">
                    {production.throughput.daily.toFixed(0)}
                  </Typography>
                  <Typography variant="caption">tonnes</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Quality Conformance
                  </Typography>
                  <Typography variant="h5">
                    {production.quality.conformance.toFixed(1)}
                  </Typography>
                  <Typography variant="caption">%</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Yield
                  </Typography>
                  <Typography variant="h5">
                    {production.quality.yield.toFixed(1)}
                  </Typography>
                  <Typography variant="caption">%</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Alerts ({furnaceAlerts.length})
              </Typography>
              <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                {furnaceAlerts.slice(0, 5).map((alert: Alert) => (
                  <ListItem key={alert.id}>
                    <ListItemIcon>
                      {alert.severity === 'critical' && <ErrorIcon color="error" />}
                      {alert.severity === 'high' && <Warning color="warning" />}
                      {alert.severity === 'medium' && <Warning color="info" />}
                    </ListItemIcon>
                    <ListItemText
                      primary={alert.title}
                      secondary={new Date(alert.timestamp).toLocaleTimeString()}
                      primaryTypographyProps={{ variant: 'body2' }}
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default OperationsDashboard;
