/**
 * Dashboard Page - Executive GHG emissions overview
 *
 * Composes 6 stat cards (Total Emissions, Scope 1, Scope 2, Scope 3,
 * YoY Change %, Data Quality Score), ScopeDonut + TrendChart in top row,
 * WaterfallChart + IntensityCard grid, TargetGauge + QualityScore,
 * and an alert feed panel.
 */

import React, { useEffect } from 'react';
import { Grid, Box, Typography, Card, CardContent, Alert, Chip } from '@mui/material';
import {
  Factory,
  ElectricBolt,
  AccountTree,
  Assessment,
  TrendingDown,
  CheckCircle,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import { fetchMetrics, fetchTrends } from '../store/slices/dashboardSlice';
import StatCard from '../components/common/StatCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ScopeDonut from '../components/dashboard/ScopeDonut';
import TrendChart from '../components/dashboard/TrendChart';
import WaterfallChart from '../components/dashboard/WaterfallChart';
import IntensityCard from '../components/dashboard/IntensityCard';
import QualityScore from '../components/dashboard/QualityScore';
import TargetGauge from '../components/dashboard/TargetGauge';
import { formatEmissions, formatChange, formatPercentage } from '../utils/formatters';

const DEMO_ORG_ID = 'demo-org';

const DashboardPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { metrics, trendData, alerts, loading, error } = useAppSelector(
    (state) => state.dashboard
  );
  const targets = useAppSelector((state) => state.targets.targets);
  const targetProgress = useAppSelector((state) => state.targets.progress);

  useEffect(() => {
    dispatch(fetchMetrics({ orgId: DEMO_ORG_ID, reportingYear: new Date().getFullYear() - 1 }));
    dispatch(fetchTrends({ orgId: DEMO_ORG_ID, startYear: new Date().getFullYear() - 5, endYear: new Date().getFullYear() - 1 }));
  }, [dispatch]);

  if (loading && !metrics) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const m = metrics;
  const scope3Categories: Record<string, number> = {};
  if (m?.top_emission_sources) {
    m.top_emission_sources.forEach((s) => {
      scope3Categories[s.source_name] = s.emissions_tco2e;
    });
  }

  return (
    <Box>
      {/* Stat cards row */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Total Emissions"
            value={m ? formatEmissions(m.total_emissions_tco2e) : '--'}
            icon={<Assessment color="primary" />}
            change={m?.year_over_year_change_percent}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Scope 1"
            value={m ? formatEmissions(m.scope1_tco2e) : '--'}
            icon={<Factory sx={{ color: '#e53935' }} />}
            color="#e53935"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Scope 2"
            value={m ? formatEmissions(m.scope2_location_tco2e) : '--'}
            subtitle="Location-based"
            icon={<ElectricBolt sx={{ color: '#1e88e5' }} />}
            color="#1e88e5"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Scope 3"
            value={m ? formatEmissions(m.scope3_tco2e) : '--'}
            icon={<AccountTree sx={{ color: '#43a047' }} />}
            color="#43a047"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="YoY Change"
            value={m ? formatChange(m.year_over_year_change_percent) : '--'}
            icon={<TrendingDown sx={{ color: m && m.year_over_year_change_percent < 0 ? '#2e7d32' : '#c62828' }} />}
            color={m && m.year_over_year_change_percent < 0 ? '#2e7d32' : '#c62828'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Data Quality"
            value={m ? formatPercentage(m.data_quality_score) : '--'}
            icon={<CheckCircle sx={{ color: '#1b5e20' }} />}
          />
        </Grid>
      </Grid>

      {/* Charts row 1: Donut + Trend */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Emissions by Scope
              </Typography>
              <ScopeDonut
                scope1={m?.scope1_tco2e ?? 0}
                scope2Location={m?.scope2_location_tco2e ?? 0}
                scope2Market={m?.scope2_market_tco2e ?? 0}
                scope3={m?.scope3_tco2e ?? 0}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Emissions Trend
              </Typography>
              <TrendChart data={trendData} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts row 2: Waterfall + Intensity */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scope 3 Category Breakdown
              </Typography>
              {Object.keys(scope3Categories).length > 0 ? (
                <WaterfallChart categories={scope3Categories} />
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                  Scope 3 category data not yet available.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {m?.top_emission_sources?.slice(0, 2).map((source, i) => (
              <Grid item xs={12} key={i}>
                <IntensityCard
                  metric={{
                    id: `intensity-${i}`,
                    inventory_id: '',
                    metric_name: source.source_name,
                    numerator_tco2e: source.emissions_tco2e,
                    denominator_value: 1,
                    denominator_unit: '',
                    intensity_value: source.emissions_tco2e,
                    intensity_unit: 'tCO2e',
                    scope_coverage: [],
                    year_over_year_change_percent: null,
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>

      {/* Bottom row: Target + Quality + Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          {targets.length > 0 ? (
            <TargetGauge
              target={targets[0]}
              currentProgress={targetProgress[targets[0].id]?.progress_percent ?? targets[0].progress_percent}
            />
          ) : (
            <Card sx={{ height: '100%' }}>
              <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 200 }}>
                <Typography variant="body2" color="text.secondary">
                  No targets set yet.
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
        <Grid item xs={12} md={4}>
          <QualityScore
            score={m?.data_quality_score ?? 0}
            dimensions={{
              completeness: m?.completeness_percent ?? 0,
              accuracy: 82,
              consistency: 88,
              timeliness: 90,
              methodology: 85,
            }}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Alerts
              </Typography>
              {alerts.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                  No alerts.
                </Typography>
              ) : (
                alerts.slice(0, 5).map((alert) => (
                  <Alert
                    key={alert.id}
                    severity={alert.severity === 'error' ? 'error' : alert.severity === 'warning' ? 'warning' : 'info'}
                    sx={{ mb: 1 }}
                  >
                    <Typography variant="body2">{alert.message}</Typography>
                  </Alert>
                ))
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
