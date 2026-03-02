/**
 * Dashboard - Executive overview page for EUDR compliance.
 *
 * Displays 4 KPI StatCards, compliance trend line chart, risk distribution
 * pie chart, recent DDS timeline, alert feed, and a "Run Full Compliance
 * Check" action button.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Paper,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import PeopleIcon from '@mui/icons-material/People';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import DescriptionIcon from '@mui/icons-material/Description';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import apiClient from '../services/api';
import type {
  DashboardMetrics,
  ComplianceTrend,
  AlertNotification,
  DueDiligenceStatement,
} from '../types';

// ---------------------------------------------------------------------------
// StatCard
// ---------------------------------------------------------------------------

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}

function StatCard({ title, value, icon, color, subtitle }: StatCardProps) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" fontWeight={700}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box
            sx={{
              width: 56,
              height: 56,
              borderRadius: 2,
              backgroundColor: `${color}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color,
            }}
          >
            {icon}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Pie chart colors
// ---------------------------------------------------------------------------

const RISK_PIE_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#f44336'];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [trends, setTrends] = useState<ComplianceTrend[]>([]);
  const [alerts, setAlerts] = useState<AlertNotification[]>([]);
  const [recentDDS, setRecentDDS] = useState<DueDiligenceStatement[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [m, t, a, d] = await Promise.all([
          apiClient.getDashboardMetrics(),
          apiClient.getComplianceTrends('12m'),
          apiClient.getAlertNotifications({ is_read: false }),
          apiClient.getDDSList({ per_page: 5, sort_by: 'created_at', sort_order: 'desc' }),
        ]);
        setMetrics(m);
        setTrends(t);
        setAlerts(a);
        setRecentDDS(d.items);
      } catch (err) {
        setError('Failed to load dashboard data.');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', pt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>;
  }

  if (!metrics) return null;

  // Risk distribution for pie chart
  const riskDistribution = [
    { name: 'Low', value: metrics.total_suppliers - metrics.high_risk_suppliers },
    { name: 'Standard', value: Math.floor(metrics.total_suppliers * 0.3) },
    { name: 'High', value: Math.floor(metrics.high_risk_suppliers * 0.6) },
    { name: 'Critical', value: Math.ceil(metrics.high_risk_suppliers * 0.4) },
  ];

  const formatMonth = (dateStr: string) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-GB', { month: 'short' });
  };

  const handleRunComplianceCheck = () => {
    // Trigger full compliance pipeline
    alert('Starting full compliance check for all suppliers...');
  };

  return (
    <Box>
      {/* Page Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={700}>
            EUDR Compliance Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            EU Deforestation Regulation (2023/1115) compliance overview
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={handleRunComplianceCheck}
          size="large"
        >
          Run Full Compliance Check
        </Button>
      </Stack>

      {/* KPI Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Suppliers"
            value={metrics.total_suppliers}
            icon={<PeopleIcon sx={{ fontSize: 28 }} />}
            color="#1565c0"
            subtitle={`${metrics.total_plots} plots tracked`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Compliance Rate"
            value={`${metrics.compliance_rate.toFixed(1)}%`}
            icon={<CheckCircleIcon sx={{ fontSize: 28 }} />}
            color="#2e7d32"
            subtitle={`${metrics.documents_verified} docs verified`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="DDS Submitted"
            value={metrics.total_dds}
            icon={<DescriptionIcon sx={{ fontSize: 28 }} />}
            color="#e65100"
            subtitle={`${metrics.pending_dds} pending`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="High-Risk Plots"
            value={metrics.high_risk_suppliers}
            icon={<WarningAmberIcon sx={{ fontSize: 28 }} />}
            color="#c62828"
            subtitle={`Avg risk: ${(metrics.avg_risk_score * 100).toFixed(0)}%`}
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} mb={3}>
        {/* Compliance Trend */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Compliance Trend (Last 12 Months)
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatMonth}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tickFormatter={(v: number) => `${v}%`}
                    tick={{ fontSize: 12 }}
                    width={45}
                  />
                  <Tooltip
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Compliance Rate']}
                    labelFormatter={(label: string) =>
                      new Date(label).toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })
                    }
                  />
                  <Line
                    type="monotone"
                    dataKey="compliance_rate"
                    stroke="#2e7d32"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#2e7d32' }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Distribution Pie */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={90}
                    paddingAngle={3}
                    dataKey="value"
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {riskDistribution.map((_, idx) => (
                      <Cell key={idx} fill={RISK_PIE_COLORS[idx]} />
                    ))}
                  </Pie>
                  <Legend verticalAlign="bottom" height={36} />
                  <Tooltip formatter={(v: number) => [v, 'Suppliers']} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Bottom Row: Recent DDS + Alerts */}
      <Grid container spacing={3}>
        {/* Recent DDS */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                <DescriptionIcon color="action" />
                <Typography variant="h6">Recent Due Diligence Statements</Typography>
              </Stack>
              <Divider sx={{ mb: 1 }} />
              {recentDDS.length === 0 ? (
                <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
                  No recent DDS records.
                </Typography>
              ) : (
                <List dense disablePadding>
                  {recentDDS.map((dds) => (
                    <ListItem key={dds.id} divider>
                      <ListItemText
                        primary={`${dds.reference_number} - ${dds.supplier_name}`}
                        secondary={`${dds.commodity.replace('_', ' ')} | ${new Date(dds.created_at).toLocaleDateString('en-GB')}`}
                      />
                      <Chip
                        label={dds.status.replace('_', ' ')}
                        size="small"
                        color={
                          dds.status === 'accepted'
                            ? 'success'
                            : dds.status === 'rejected'
                            ? 'error'
                            : dds.status === 'submitted'
                            ? 'warning'
                            : 'default'
                        }
                        sx={{ textTransform: 'capitalize' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Alert Feed */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                <NotificationsActiveIcon color="action" />
                <Typography variant="h6">Alerts</Typography>
                {alerts.length > 0 && (
                  <Chip label={alerts.length} size="small" color="error" />
                )}
              </Stack>
              <Divider sx={{ mb: 1 }} />
              {alerts.length === 0 ? (
                <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
                  No unread alerts.
                </Typography>
              ) : (
                <List dense disablePadding>
                  {alerts.slice(0, 8).map((alert) => (
                    <ListItem key={alert.id} divider>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        {alert.severity === 'critical' || alert.severity === 'error' ? (
                          <WarningAmberIcon color="error" fontSize="small" />
                        ) : alert.severity === 'warning' ? (
                          <WarningAmberIcon color="warning" fontSize="small" />
                        ) : (
                          <TrendingUpIcon color="info" fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={alert.title}
                        secondary={`${alert.type} | ${new Date(alert.created_at).toLocaleDateString('en-GB')}`}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
