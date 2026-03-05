/**
 * Dashboard - Executive overview of TCFD climate disclosure status.
 *
 * Displays KPI stat cards, risk/opportunity charts, scenario comparison,
 * disclosure completion donut, emissions summary, and year-over-year trends.
 */

import React, { useEffect, useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Alert, Chip, List, ListItem, ListItemText } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line,
} from 'recharts';
import {
  Warning, EmojiObjects, Assessment, TrendingDown, Thermostat, Description,
} from '@mui/icons-material';
import StatCard from '../components/common/StatCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDashboardSummary, fetchEmissionsTrend, selectDashboardSummary, selectEmissionsTrend, selectDashboardLoading } from '../store/slices/dashboardSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const RISK_COLORS = ['#B71C1C', '#E65100', '#F57F17', '#1B5E20', '#388E3C'];
const PIE_COLORS = ['#1B5E20', '#4CAF50', '#81C784', '#C8E6C9', '#EF6C00', '#E0E0E0'];
const SCENARIO_COLORS = ['#0D47A1', '#1B5E20', '#E65100', '#7B1FA2'];

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

  // Demo data for when API hasn't returned yet
  const demoSummary = useMemo(() => ({
    risk_exposure: {
      total_financial_impact: 45200000,
      physical_risk_total: 18500000,
      transition_risk_total: 26700000,
      risk_count_by_level: { critical: 2, high: 5, medium: 8, low: 12, negligible: 3 } as Record<string, number>,
      top_risks: [
        { id: '1', name: 'Carbon pricing regulation', level: 'high' as const, impact: 12500000 },
        { id: '2', name: 'Extreme weather events', level: 'critical' as const, impact: 8300000 },
        { id: '3', name: 'Technology disruption', level: 'medium' as const, impact: 6200000 },
      ],
    },
    opportunity_value: {
      total_opportunity_value: 62800000,
      total_cost_savings: 8400000,
      opportunity_count_by_type: { resource_efficiency: 3, energy_source: 4, products_services: 5, markets: 2, resilience: 1 } as Record<string, number>,
      top_opportunities: [
        { id: '1', name: 'Green product line expansion', value: 25000000, status: 'evaluating' },
        { id: '2', name: 'Renewable energy procurement', value: 15000000, status: 'implementing' },
      ],
    },
    disclosure_maturity: {
      overall_pct: 68,
      pillar_scores: { governance: 82, strategy: 65, risk_management: 70, metrics_targets: 55 } as Record<string, number>,
      section_statuses: [
        { code: 'GOV-A', title: 'Board Oversight', status: 'final' },
        { code: 'GOV-B', title: 'Management Role', status: 'review' },
        { code: 'STR-A', title: 'Risks & Opportunities', status: 'draft' },
        { code: 'STR-B', title: 'Business Impact', status: 'in_progress' },
        { code: 'STR-C', title: 'Resilience', status: 'draft' },
        { code: 'RM-A', title: 'Risk Identification', status: 'review' },
        { code: 'RM-B', title: 'Risk Management Process', status: 'draft' },
        { code: 'RM-C', title: 'Integration', status: 'in_progress' },
        { code: 'MT-A', title: 'Climate Metrics', status: 'draft' },
        { code: 'MT-B', title: 'GHG Emissions', status: 'review' },
        { code: 'MT-C', title: 'Targets', status: 'not_started' },
      ],
    },
    scenario_summary: {
      scenarios_analyzed: 3,
      net_impact_range: { low: -32000000, high: 18000000 },
      key_driver: 'Carbon price trajectory',
      scenario_results: [
        { name: 'Net Zero 2050', net_impact: -12500000 },
        { name: 'Announced Pledges', net_impact: -5800000 },
        { name: 'Current Policies', net_impact: -28400000 },
      ],
    },
  }), []);

  const demoTrend = useMemo(() => [
    { year: 2020, scope_1: 42000, scope_2: 28000, scope_3: 185000, total: 255000 },
    { year: 2021, scope_1: 39500, scope_2: 25000, scope_3: 178000, total: 242500 },
    { year: 2022, scope_1: 37000, scope_2: 22000, scope_3: 170000, total: 229000 },
    { year: 2023, scope_1: 34500, scope_2: 19500, scope_3: 162000, total: 216000 },
    { year: 2024, scope_1: 31800, scope_2: 17000, scope_3: 155000, total: 203800 },
    { year: 2025, scope_1: 29000, scope_2: 14500, scope_3: 148000, total: 191500 },
  ], []);

  const data = summary || demoSummary;
  const trend = emissionsTrend.length > 0 ? emissionsTrend : demoTrend;

  if (loading && !summary) return <LoadingSpinner message="Loading dashboard..." />;

  const riskByLevel = Object.entries(data.risk_exposure.risk_count_by_level).map(([level, count]) => ({
    level: level.charAt(0).toUpperCase() + level.slice(1),
    count,
  }));

  const disclosureData = [
    { name: 'Final', value: data.disclosure_maturity.section_statuses.filter((s) => s.status === 'final' || s.status === 'published').length },
    { name: 'Review', value: data.disclosure_maturity.section_statuses.filter((s) => s.status === 'review').length },
    { name: 'Draft', value: data.disclosure_maturity.section_statuses.filter((s) => s.status === 'draft').length },
    { name: 'In Progress', value: data.disclosure_maturity.section_statuses.filter((s) => s.status === 'in_progress').length },
    { name: 'Not Started', value: data.disclosure_maturity.section_statuses.filter((s) => s.status === 'not_started').length },
  ].filter((d) => d.value > 0);

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3 }}>TCFD Climate Disclosure Dashboard</Typography>

      {/* KPI Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Risk Exposure"
            value={data.risk_exposure.total_financial_impact}
            format="currency"
            icon={<Warning />}
            trend={-8.2}
            trendLabel="vs last year"
            color="error"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Opportunity Value"
            value={data.opportunity_value.total_opportunity_value}
            format="currency"
            icon={<EmojiObjects />}
            trend={12.5}
            trendLabel="vs last year"
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Disclosure Maturity"
            value={data.disclosure_maturity.overall_pct}
            format="percent"
            icon={<Description />}
            trend={15}
            trendLabel="vs last year"
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Scenarios Analyzed"
            value={data.scenario_summary.scenarios_analyzed}
            icon={<Thermostat />}
            subtitle={`Key driver: ${data.scenario_summary.key_driver}`}
            color="secondary"
          />
        </Grid>
      </Grid>

      {/* Charts Row 1 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Risk by Level */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Risk Distribution by Level</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={riskByLevel}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" fontSize={12} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Risks">
                    {riskByLevel.map((_, idx) => (
                      <Cell key={idx} fill={RISK_COLORS[idx % RISK_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Scenario Comparison */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Scenario Impact Comparison</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data.scenario_summary.scenario_results} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={(v) => `$${(v / 1e6).toFixed(0)}M`} />
                  <YAxis type="category" dataKey="name" fontSize={11} width={110} />
                  <Tooltip formatter={(value: number) => [`$${(value / 1e6).toFixed(1)}M`, 'Net Impact']} />
                  <Bar dataKey="net_impact" name="Net Impact">
                    {data.scenario_summary.scenario_results.map((entry, idx) => (
                      <Cell key={idx} fill={entry.net_impact >= 0 ? '#2E7D32' : '#C62828'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Disclosure Donut */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Disclosure Completion</Typography>
              <Box sx={{ position: 'relative' }}>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={disclosureData}
                      cx="50%"
                      cy="50%"
                      innerRadius={70}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {disclosureData.map((_, idx) => (
                        <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <Box sx={{
                  position: 'absolute', top: '50%', left: '50%',
                  transform: 'translate(-50%, -50%)', textAlign: 'center',
                }}>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>{data.disclosure_maturity.overall_pct}%</Typography>
                  <Typography variant="caption" color="text.secondary">Complete</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Row 2 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Emissions Trend */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emissions Trend (tCO2e)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value: number) => [value.toLocaleString(), '']} />
                  <Legend />
                  <Line type="monotone" dataKey="scope_1" stroke="#1B5E20" strokeWidth={2} name="Scope 1" />
                  <Line type="monotone" dataKey="scope_2" stroke="#0D47A1" strokeWidth={2} name="Scope 2" />
                  <Line type="monotone" dataKey="scope_3" stroke="#EF6C00" strokeWidth={2} name="Scope 3" />
                  <Line type="monotone" dataKey="total" stroke="#B71C1C" strokeWidth={2} strokeDasharray="5 5" name="Total" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Pillar Scores */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Pillar Maturity Scores</Typography>
              {Object.entries(data.disclosure_maturity.pillar_scores).map(([pillar, score]) => (
                <Box key={pillar} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize', fontWeight: 500 }}>
                      {pillar.replace(/_/g, ' ')}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{score}%</Typography>
                  </Box>
                  <Box sx={{
                    height: 8, borderRadius: 4, backgroundColor: '#E0E0E0',
                    overflow: 'hidden',
                  }}>
                    <Box sx={{
                      height: '100%', borderRadius: 4, width: `${score}%`,
                      backgroundColor: score >= 75 ? '#2E7D32' : score >= 50 ? '#EF6C00' : '#C62828',
                    }} />
                  </Box>
                </Box>
              ))}

              <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>Top Risks</Typography>
              <List dense>
                {data.risk_exposure.top_risks.slice(0, 3).map((risk) => (
                  <ListItem key={risk.id} disableGutters>
                    <ListItemText
                      primary={risk.name}
                      secondary={`$${(risk.impact / 1e6).toFixed(1)}M`}
                      primaryTypographyProps={{ fontSize: '0.8rem' }}
                      secondaryTypographyProps={{ fontSize: '0.75rem' }}
                    />
                    <Chip
                      label={risk.level}
                      size="small"
                      sx={{
                        backgroundColor: risk.level === 'critical' ? '#B71C1C' : risk.level === 'high' ? '#E65100' : '#F57F17',
                        color: 'white',
                        fontSize: '0.7rem',
                        textTransform: 'capitalize',
                      }}
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

export default Dashboard;
