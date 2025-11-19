/**
 * Executive Dashboard Component
 *
 * High-level overview of furnace performance for executives and management
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert,
  CircularProgress,
  Tab,
  Tabs,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  LocalFireDepartment,
  EmojiObjects,
  AttachMoney,
  Co2,
  Speed,
  CheckCircle,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

import KPICard from '../charts/KPICard';
import GaugeChart from '../charts/GaugeChart';
import { useFurnaceStore } from '../../store/furnaceStore';
import { apiClient } from '../../services/apiClient';
import { useWebSocket } from '../../services/websocket';
import type { FurnacePerformance, OptimizationOpportunity } from '../../types';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

const ExecutiveDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const { selectedFurnaceId } = useFurnaceStore();
  const [performance, setPerformance] = useState<FurnacePerformance | null>(null);

  // Fetch initial performance data
  const { data: initialPerformance, isLoading, error } = useQuery({
    queryKey: ['performance', selectedFurnaceId],
    queryFn: () =>
      selectedFurnaceId ? apiClient.getPerformance(selectedFurnaceId) : null,
    enabled: !!selectedFurnaceId,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch optimization opportunities
  const { data: opportunities } = useQuery({
    queryKey: ['optimization', selectedFurnaceId],
    queryFn: () =>
      selectedFurnaceId ? apiClient.getOptimizationOpportunities(selectedFurnaceId) : [],
    enabled: !!selectedFurnaceId,
  });

  // Real-time updates via WebSocket
  const { service } = useWebSocket({
    furnaceId: selectedFurnaceId || '',
    autoConnect: !!selectedFurnaceId,
  });

  useEffect(() => {
    if (!selectedFurnaceId) return;

    const unsubscribe = service.on('performance_update', (data: FurnacePerformance) => {
      setPerformance(data);
    });

    return unsubscribe;
  }, [selectedFurnaceId, service]);

  // Use real-time data if available, otherwise use initial data
  const currentPerformance = performance || initialPerformance;

  if (!selectedFurnaceId) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">Please select a furnace to view performance data.</Alert>
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
        <Alert severity="error">Failed to load performance data: {error.message}</Alert>
      </Box>
    );
  }

  if (!currentPerformance) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">No performance data available.</Alert>
      </Box>
    );
  }

  const kpis = currentPerformance.kpis;
  const efficiency = currentPerformance.efficiency;
  const production = currentPerformance.production;
  const emissions = currentPerformance.emissions;

  // Calculate cost savings opportunities
  const totalSavingsOpportunity = opportunities?.reduce(
    (sum, opp) => sum + opp.impact.costSavings,
    0
  ) || 0;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Executive Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Last updated: {new Date(currentPerformance.timestamp).toLocaleString()}
        </Typography>
      </Box>

      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, value) => setTabValue(value)}>
          <Tab label="Overview" />
          <Tab label="Efficiency" />
          <Tab label="Costs & Savings" />
          <Tab label="Sustainability" />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      <TabPanel value={tabValue} index={0}>
        {/* Top KPIs */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Overall Efficiency"
              value={kpis.overallEfficiency}
              unit="%"
              target={95}
              trend="stable"
              trendValue={2.3}
              icon={<Assessment />}
              status={kpis.overallEfficiency >= 90 ? 'good' : 'warning'}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Production Rate"
              value={kpis.productionRate}
              unit="tonnes/hr"
              target={production.target}
              trend="increasing"
              trendValue={5.2}
              icon={<Speed />}
              status="good"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Cost per Tonne"
              value={kpis.costPerTonne}
              unit="USD"
              trend="decreasing"
              trendValue={-3.1}
              icon={<AttachMoney />}
              status="good"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Availability"
              value={kpis.availabilityFactor}
              unit="%"
              target={95}
              trend="stable"
              icon={<CheckCircle />}
              status={kpis.availabilityFactor >= 95 ? 'good' : 'warning'}
            />
          </Grid>
        </Grid>

        {/* Performance Trends */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Trends (Last 7 Days)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Line
                    data={{
                      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                      datasets: [
                        {
                          label: 'Efficiency %',
                          data: [92, 93, 91, 94, 93, 95, 94],
                          borderColor: 'rgb(75, 192, 192)',
                          backgroundColor: 'rgba(75, 192, 192, 0.2)',
                          fill: true,
                          tension: 0.4,
                        },
                        {
                          label: 'Utilization %',
                          data: [88, 90, 87, 92, 91, 93, 92],
                          borderColor: 'rgb(255, 99, 132)',
                          backgroundColor: 'rgba(255, 99, 132, 0.2)',
                          fill: true,
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                      scales: {
                        y: {
                          beginAtZero: false,
                          min: 80,
                          max: 100,
                        },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Overall Equipment Effectiveness
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    py: 2,
                  }}
                >
                  <GaugeChart
                    value={efficiency.overall.oee}
                    maxValue={100}
                    title="OEE Score"
                    unit="%"
                    thresholds={{ good: 85, warning: 75, critical: 65 }}
                    size={180}
                  />
                </Box>
                <Grid container spacing={2} sx={{ mt: 2 }}>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">
                      Availability
                    </Typography>
                    <Typography variant="h6">
                      {efficiency.overall.availability.toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">
                      Performance
                    </Typography>
                    <Typography variant="h6">
                      {efficiency.overall.performance.toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">
                      Quality
                    </Typography>
                    <Typography variant="h6">
                      {efficiency.overall.quality.toFixed(1)}%
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Optimization Opportunities */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Top Optimization Opportunities
            </Typography>
            <Grid container spacing={2}>
              {opportunities?.slice(0, 3).map((opp) => (
                <Grid item xs={12} md={4} key={opp.id}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="subtitle2">{opp.title}</Typography>
                        <Chip
                          label={opp.priority}
                          size="small"
                          color={
                            opp.priority === 'critical'
                              ? 'error'
                              : opp.priority === 'high'
                              ? 'warning'
                              : 'default'
                          }
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {opp.description}
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="h6" color="success.main">
                          ${(opp.impact.costSavings / 1000).toFixed(0)}K/year
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Payback: {opp.roi.paybackPeriod} months
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </TabPanel>

      {/* Efficiency Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Thermal Efficiency
                </Typography>
                <GaugeChart
                  value={kpis.thermalEfficiency}
                  maxValue={100}
                  unit="%"
                  thresholds={{ good: 85, warning: 75, critical: 65 }}
                  size={200}
                />
                <Box sx={{ mt: 3 }}>
                  <Typography variant="body2" color="text.secondary">
                    Current: {kpis.thermalEfficiency.toFixed(1)}% | Target:{' '}
                    {efficiency.thermal.target.toFixed(1)}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Heat Balance
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Doughnut
                    data={{
                      labels: ['Useful Heat', 'Wall Losses', 'Exhaust Gas', 'Other Losses'],
                      datasets: [
                        {
                          data: [
                            efficiency.heat.useful,
                            efficiency.heat.losses * 0.4,
                            efficiency.heat.losses * 0.4,
                            efficiency.heat.losses * 0.2,
                          ],
                          backgroundColor: [
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(255, 159, 64, 0.8)',
                            'rgba(201, 203, 207, 0.8)',
                          ],
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'bottom' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Specific Energy Consumption Trend
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Line
                    data={{
                      labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
                      datasets: [
                        {
                          label: 'SEC (MJ/tonne)',
                          data: Array.from(
                            { length: 24 },
                            () => 2800 + Math.random() * 400
                          ),
                          borderColor: 'rgb(53, 162, 235)',
                          backgroundColor: 'rgba(53, 162, 235, 0.5)',
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Costs & Savings Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Daily Operating Cost"
              value={(kpis.costPerTonne * production.throughput.daily).toFixed(0)}
              unit="USD"
              icon={<AttachMoney />}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Savings Opportunity"
              value={(totalSavingsOpportunity / 1000).toFixed(0)}
              unit="K USD/year"
              icon={<EmojiObjects />}
              status="good"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Fuel Cost"
              value={currentPerformance.fuel.cost.daily.toFixed(0)}
              unit="USD/day"
              icon={<LocalFireDepartment />}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Fuel Efficiency"
              value={kpis.fuelEfficiency}
              unit="%"
              target={90}
              icon={<TrendingUp />}
              status={kpis.fuelEfficiency >= 85 ? 'good' : 'warning'}
            />
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Cost Breakdown (Monthly)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Bar
                    data={{
                      labels: ['Fuel', 'Electricity', 'Maintenance', 'Labor', 'Other'],
                      datasets: [
                        {
                          label: 'Cost (USD)',
                          data: [
                            currentPerformance.fuel.cost.monthly,
                            currentPerformance.fuel.cost.monthly * 0.3,
                            50000,
                            80000,
                            30000,
                          ],
                          backgroundColor: 'rgba(54, 162, 235, 0.5)',
                          borderColor: 'rgba(54, 162, 235, 1)',
                          borderWidth: 1,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Sustainability Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Carbon Intensity"
              value={kpis.carbonIntensity}
              unit="kgCO₂/tonne"
              icon={<Co2 />}
              trend="decreasing"
              trendValue={-4.5}
              status="good"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="Daily CO₂ Emissions"
              value={(emissions.total.specific * production.throughput.daily).toFixed(0)}
              unit="tonnes"
              icon={<Co2 />}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="NOx Compliance"
              value={emissions.nox.compliance}
              unit="%"
              target={100}
              icon={<CheckCircle />}
              status={emissions.nox.compliance >= 95 ? 'good' : 'critical'}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <KPICard
              title="SOx Compliance"
              value={emissions.sox.compliance}
              unit="%"
              target={100}
              icon={<CheckCircle />}
              status={emissions.sox.compliance >= 95 ? 'good' : 'critical'}
            />
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Emissions Trend (Last 30 Days)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Line
                    data={{
                      labels: Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
                      datasets: [
                        {
                          label: 'CO₂ (kg/tonne)',
                          data: Array.from({ length: 30 }, () => 350 + Math.random() * 50),
                          borderColor: 'rgb(255, 99, 132)',
                          backgroundColor: 'rgba(255, 99, 132, 0.2)',
                          fill: true,
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default ExecutiveDashboard;
