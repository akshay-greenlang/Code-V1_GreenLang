/**
 * Dashboard Page - Executive ISO 14064-1 overview
 *
 * Composes KPI cards (gross/removals/net/YoY%), category breakdown chart,
 * gas breakdown chart, trend line chart, verification status, and alerts.
 */

import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Grid,
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  Chip,
} from '@mui/material';
import {
  Assessment,
  Park,
  BarChart as BarChartIcon,
  TrendingDown,
  TrendingUp,
  CheckCircle,
  VerifiedUser,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import type { AppDispatch, AppRootState } from '../store';
import { fetchMetrics, fetchTrends, fetchAlerts } from '../store/slices/dashboardSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import GasBreakdownChart from '../components/quantification/GasBreakdownChart';
import StatusChip from '../components/common/StatusChip';
import {
  CATEGORY_COLORS,
  ISOCategory,
  ISO_CATEGORY_SHORT_NAMES,
} from '../types';
import { formatTCO2e, formatPercentage, formatNumber } from '../utils/formatters';

const DEMO_ORG_ID = 'demo-org';
const REPORTING_YEAR = new Date().getFullYear() - 1;

interface StatCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: React.ReactNode;
  color?: string;
  change?: number | null;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle, icon, color, change }) => (
  <Card>
    <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        {icon}
        <Typography variant="caption" color="text.secondary">
          {title}
        </Typography>
      </Box>
      <Typography variant="h5" fontWeight={700} sx={{ color: color || '#1a1a2e' }}>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="caption" color="text.secondary">
          {subtitle}
        </Typography>
      )}
      {change != null && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
          {change < 0 ? (
            <TrendingDown fontSize="small" sx={{ color: '#2e7d32' }} />
          ) : (
            <TrendingUp fontSize="small" sx={{ color: '#c62828' }} />
          )}
          <Typography
            variant="caption"
            color={change < 0 ? 'success.main' : 'error.main'}
            fontWeight={600}
          >
            {formatPercentage(change)} vs prior year
          </Typography>
        </Box>
      )}
    </CardContent>
  </Card>
);

const DashboardPage: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { metrics, trendData, alerts, loading, error } = useSelector(
    (s: AppRootState) => s.dashboard,
  );

  useEffect(() => {
    dispatch(fetchMetrics({ orgId: DEMO_ORG_ID, reportingYear: REPORTING_YEAR }));
    dispatch(
      fetchTrends({
        orgId: DEMO_ORG_ID,
        startYear: REPORTING_YEAR - 4,
        endYear: REPORTING_YEAR,
      }),
    );
    dispatch(fetchAlerts(DEMO_ORG_ID));
  }, [dispatch]);

  if (loading && !metrics) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const m = metrics;

  // Prepare category breakdown chart data
  const catChartData = m?.by_category
    ? Object.entries(m.by_category).map(([cat, value]) => ({
        name: ISO_CATEGORY_SHORT_NAMES[cat as ISOCategory] || cat,
        value,
        fill: CATEGORY_COLORS[cat as ISOCategory] || '#9e9e9e',
      }))
    : [];

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        ISO 14064-1 Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Reporting Year: {REPORTING_YEAR}
      </Typography>

      {/* KPI Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Gross Emissions"
            value={formatTCO2e(m?.gross_emissions_tco2e)}
            icon={<Assessment sx={{ color: '#e53935' }} />}
            color="#e53935"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Removals"
            value={formatTCO2e(m?.total_removals_tco2e)}
            icon={<Park sx={{ color: '#1b5e20' }} />}
            color="#1b5e20"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Net Emissions"
            value={formatTCO2e(m?.net_emissions_tco2e)}
            icon={<BarChartIcon sx={{ color: '#1e88e5' }} />}
            color="#1e88e5"
            change={m?.yoy_change_pct}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Data Quality"
            value={m ? `${m.data_quality_score.toFixed(0)}%` : '--%'}
            subtitle={`${m?.completeness_pct.toFixed(0) ?? 0}% complete`}
            icon={<CheckCircle sx={{ color: '#1b5e20' }} />}
            color="#1b5e20"
          />
        </Grid>
      </Grid>

      {/* Charts row 1: Category breakdown + Gas breakdown */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Emissions by Category
              </Typography>
              {catChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={catChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                    <YAxis />
                    <Tooltip
                      formatter={(value: number) => [`${formatNumber(value, 2)} tCO2e`]}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {catChartData.map((entry, idx) => (
                        <Bar key={idx} dataKey="value" fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
                  No category data available.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Gas Breakdown
              </Typography>
              <GasBreakdownChart
                data={m?.by_gas ?? {}}
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts row 2: Trend + Verification + Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Emissions Trend
              </Typography>
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period_label" />
                    <YAxis />
                    <Tooltip
                      formatter={(value: number, name: string) => [
                        `${formatNumber(value, 2)} tCO2e`,
                        name.replace(/_/g, ' '),
                      ]}
                    />
                    <Legend
                      formatter={(value: string) => value.replace(/_/g, ' ')}
                    />
                    <Line
                      type="monotone"
                      dataKey="gross_total_tco2e"
                      name="Gross Total"
                      stroke="#e53935"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="net_total_tco2e"
                      name="Net Total"
                      stroke="#1e88e5"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="removals_tco2e"
                      name="Removals"
                      stroke="#1b5e20"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
                  No trend data available.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {/* Verification status card */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <VerifiedUser sx={{ color: '#1b5e20' }} />
                    <Typography variant="h6">Verification</Typography>
                  </Box>
                  <StatusChip status={m?.verification_stage || 'draft'} />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {m?.management_plan_actions ?? 0} management actions defined
                  </Typography>
                  {m?.significant_categories && m.significant_categories.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Significant categories:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                        {m.significant_categories.map((cat) => (
                          <Chip
                            key={cat}
                            label={ISO_CATEGORY_SHORT_NAMES[cat as ISOCategory] || cat}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Alerts */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Alerts
                  </Typography>
                  {alerts.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
                      No alerts.
                    </Typography>
                  ) : (
                    alerts.slice(0, 5).map((alert) => (
                      <Alert
                        key={alert.id}
                        severity={
                          alert.severity === 'error'
                            ? 'error'
                            : alert.severity === 'warning'
                            ? 'warning'
                            : 'info'
                        }
                        sx={{ mb: 1 }}
                      >
                        <Typography variant="body2" fontWeight={500}>
                          {alert.title}
                        </Typography>
                        <Typography variant="caption">{alert.message}</Typography>
                      </Alert>
                    ))
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
