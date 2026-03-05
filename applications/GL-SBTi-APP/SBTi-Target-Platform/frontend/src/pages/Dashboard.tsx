/**
 * Dashboard - Executive overview of SBTi target validation & progress status.
 *
 * Displays readiness score, target status cards, pathway preview chart,
 * temperature gauge, review countdown, and milestone timeline.
 */

import React, { useEffect, useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Alert, Chip, List, ListItem, ListItemText } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
} from 'recharts';
import {
  GpsFixed, TrendingDown, Thermostat, CheckCircle, Schedule, TrackChanges,
} from '@mui/icons-material';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ScoreGauge from '../components/common/ScoreGauge';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDashboardSummary, fetchEmissionsTrend, selectDashboardSummary, selectEmissionsTrend, selectDashboardLoading } from '../store/slices/dashboardSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const STATUS_COLORS: Record<string, string> = { draft: '#9E9E9E', submitted: '#0D47A1', validated: '#2E7D32', active: '#1B5E20', expired: '#C62828' };
const SCOPE_COLORS = ['#1B5E20', '#0D47A1', '#EF6C00'];
const PIE_COLORS = ['#1B5E20', '#4CAF50', '#81C784', '#C8E6C9', '#EF6C00', '#E0E0E0'];

const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const summary = useAppSelector(selectDashboardSummary);
  const emissionsTrend = useAppSelector(selectEmissionsTrend);
  const loading = useAppSelector(selectDashboardLoading);

  useEffect(() => {
    dispatch(fetchDashboardSummary(orgId));
    dispatch(fetchEmissionsTrend(orgId));
  }, [dispatch, orgId]);

  /* Demo data for when API hasn't returned yet */
  const demoSummary = useMemo(() => ({
    readiness_score: 72,
    total_targets: 6,
    active_targets: 4,
    near_term_targets: 3,
    long_term_targets: 2,
    net_zero_targets: 1,
    overall_progress_pct: 35,
    temperature_score: 1.8,
    scope3_coverage_pct: 68,
    validation_status: 'submitted' as const,
    next_review_date: '2029-06-15',
    days_to_review: 1199,
    recent_activities: [
      { id: '1', action: 'Scope 1+2 near-term target submitted', date: '2025-02-28', actor: 'Sarah Chen' },
      { id: '2', action: 'SDA pathway recalculated for Power sector', date: '2025-02-25', actor: 'James Mitchell' },
      { id: '3', action: 'Scope 3 screening trigger assessment completed', date: '2025-02-20', actor: 'System' },
      { id: '4', action: 'Annual progress recorded for FY2024', date: '2025-02-15', actor: 'Aisha Rahman' },
    ],
    targets_by_status: { draft: 1, submitted: 2, validated: 1, active: 2, expired: 0 } as Record<string, number>,
    pathway_alignment: '1.5C' as const,
  }), []);

  const demoTrend = useMemo(() => [
    { year: 2019, scope_1: 55000, scope_2: 35000, scope_3: 210000, total: 300000, pathway: 300000 },
    { year: 2020, scope_1: 52000, scope_2: 32000, scope_3: 200000, total: 284000, pathway: 287400 },
    { year: 2021, scope_1: 49000, scope_2: 29000, scope_3: 190000, total: 268000, pathway: 274800 },
    { year: 2022, scope_1: 46000, scope_2: 26000, scope_3: 183000, total: 255000, pathway: 262200 },
    { year: 2023, scope_1: 43000, scope_2: 23000, scope_3: 175000, total: 241000, pathway: 249600 },
    { year: 2024, scope_1: 40000, scope_2: 20000, scope_3: 168000, total: 228000, pathway: 237000 },
    { year: 2025, scope_1: 37500, scope_2: 18000, scope_3: 162000, total: 217500, pathway: 224400 },
  ], []);

  const data = summary || demoSummary;
  const trend = emissionsTrend.length > 0 ? emissionsTrend : demoTrend;

  if (loading && !summary) return <LoadingSpinner message="Loading dashboard..." />;

  const targetsByStatus = Object.entries(data.targets_by_status).filter(([, v]) => v > 0).map(([k, v]) => ({
    name: k.charAt(0).toUpperCase() + k.slice(1), value: v,
  }));

  const tempColor = data.temperature_score <= 1.5 ? '#1B5E20' : data.temperature_score <= 1.8 ? '#2E7D32' : data.temperature_score <= 2.0 ? '#EF6C00' : '#C62828';

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3 }}>SBTi Target Platform Dashboard</Typography>

      {/* KPI Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <TrackChanges color="primary" />
                <Typography variant="subtitle2" color="text.secondary">Readiness Score</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 1 }}>
                <ScoreGauge value={data.readiness_score} size={100} />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
                {data.readiness_score >= 80 ? 'Submission ready' : 'Gaps remain'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <GpsFixed color="primary" />
                <Typography variant="subtitle2" color="text.secondary">Active Targets</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 700, textAlign: 'center', my: 1 }}>
                {data.active_targets} / {data.total_targets}
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
                <Chip label={`${data.near_term_targets} Near-term`} size="small" sx={{ fontSize: '0.65rem', backgroundColor: '#E8F5E9' }} />
                <Chip label={`${data.long_term_targets} Long-term`} size="small" sx={{ fontSize: '0.65rem', backgroundColor: '#E3F2FD' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Thermostat sx={{ color: tempColor }} />
                <Typography variant="subtitle2" color="text.secondary">Temperature Score</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 700, textAlign: 'center', my: 1, color: tempColor }}>
                {data.temperature_score.toFixed(1)}&deg;C
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
                Pathway: {data.pathway_alignment}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Schedule color="primary" />
                <Typography variant="subtitle2" color="text.secondary">Next 5-Year Review</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 700, textAlign: 'center', my: 1 }}>
                {data.days_to_review}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
                days remaining ({data.next_review_date})
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Row 1 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Emissions vs Pathway */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emissions vs Pathway (tCO2e)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value: number) => [value.toLocaleString(), '']} />
                  <Legend />
                  <Area type="monotone" dataKey="scope_1" stackId="1" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.6} name="Scope 1" />
                  <Area type="monotone" dataKey="scope_2" stackId="1" stroke="#0D47A1" fill="#0D47A1" fillOpacity={0.6} name="Scope 2" />
                  <Area type="monotone" dataKey="scope_3" stackId="1" stroke="#EF6C00" fill="#EF6C00" fillOpacity={0.4} name="Scope 3" />
                  <Line type="monotone" dataKey="pathway" stroke="#C62828" strokeWidth={2} strokeDasharray="5 5" name="Pathway Target" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Target Status Distribution */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Target Status Distribution</Typography>
              <Box sx={{ position: 'relative' }}>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={targetsByStatus}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {targetsByStatus.map((entry, idx) => (
                        <Cell key={idx} fill={STATUS_COLORS[entry.name.toLowerCase()] || PIE_COLORS[idx % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>{data.total_targets}</Typography>
                  <Typography variant="caption" color="text.secondary">Targets</Typography>
                </Box>
              </Box>

              {/* Progress Bar */}
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>Overall Progress</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{data.overall_progress_pct}%</Typography>
                </Box>
                <Box sx={{ height: 8, borderRadius: 4, backgroundColor: '#E0E0E0', overflow: 'hidden' }}>
                  <Box sx={{
                    height: '100%', borderRadius: 4, width: `${data.overall_progress_pct}%`,
                    backgroundColor: data.overall_progress_pct >= 75 ? '#2E7D32' : data.overall_progress_pct >= 40 ? '#EF6C00' : '#C62828',
                  }} />
                </Box>
              </Box>

              {/* Scope 3 Coverage */}
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>Scope 3 Coverage</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{data.scope3_coverage_pct}%</Typography>
                </Box>
                <Box sx={{ height: 8, borderRadius: 4, backgroundColor: '#E0E0E0', overflow: 'hidden' }}>
                  <Box sx={{
                    height: '100%', borderRadius: 4, width: `${data.scope3_coverage_pct}%`,
                    backgroundColor: data.scope3_coverage_pct >= 67 ? '#2E7D32' : data.scope3_coverage_pct >= 40 ? '#EF6C00' : '#C62828',
                  }} />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {data.scope3_coverage_pct >= 67 ? 'Meets 2/3 threshold' : 'Below 2/3 threshold'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activities */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Recent Activities</Typography>
              <List dense>
                {data.recent_activities.slice(0, 5).map((activity) => (
                  <ListItem key={activity.id} disableGutters sx={{ borderBottom: '1px solid #F0F0F0' }}>
                    <ListItemText
                      primary={activity.action}
                      secondary={`${activity.date} by ${activity.actor}`}
                      primaryTypographyProps={{ fontSize: '0.85rem' }}
                      secondaryTypographyProps={{ fontSize: '0.75rem' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Scope Emissions Trend (tCO2e)</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value: number) => [value.toLocaleString(), '']} />
                  <Legend />
                  <Line type="monotone" dataKey="scope_1" stroke={SCOPE_COLORS[0]} strokeWidth={2} name="Scope 1" />
                  <Line type="monotone" dataKey="scope_2" stroke={SCOPE_COLORS[1]} strokeWidth={2} name="Scope 2" />
                  <Line type="monotone" dataKey="scope_3" stroke={SCOPE_COLORS[2]} strokeWidth={2} name="Scope 3" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
