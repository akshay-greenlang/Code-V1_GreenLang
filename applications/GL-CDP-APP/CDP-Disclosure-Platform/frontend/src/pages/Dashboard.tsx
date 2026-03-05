/**
 * Dashboard Page - CDP Executive Dashboard
 *
 * Composes ScoreCard, ModuleProgress, GapSummary, TimelineCountdown,
 * CategoryRadar, ReadinessScore, TrendChart, AListChecklist, and alerts.
 * Fetches all dashboard data on mount via Redux thunks.
 */

import React, { useEffect } from 'react';
import { Grid, Box, Typography, Alert, Card, CardContent } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchDashboard,
  fetchDashboardAlerts,
  markAlertRead,
} from '../store/slices/dashboardSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ScoreCard from '../components/dashboard/ScoreCard';
import ModuleProgressComponent from '../components/dashboard/ModuleProgress';
import GapSummary from '../components/dashboard/GapSummary';
import TimelineCountdown from '../components/dashboard/TimelineCountdown';
import CategoryRadar from '../components/dashboard/CategoryRadar';
import ReadinessScore from '../components/dashboard/ReadinessScore';
import TrendChart from '../components/dashboard/TrendChart';
import AListChecklist from '../components/dashboard/AListChecklist';

const DEMO_ORG_ID = 'demo-org';
const REPORTING_YEAR = new Date().getFullYear();

const DashboardPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { data, alerts, loading, error } = useAppSelector((s) => s.dashboard);

  useEffect(() => {
    dispatch(fetchDashboard({ orgId: DEMO_ORG_ID, reportingYear: REPORTING_YEAR }));
    dispatch(fetchDashboardAlerts(DEMO_ORG_ID));
  }, [dispatch]);

  if (loading && !data) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return null;

  const handleAlertRead = (alertId: string) => {
    dispatch(markAlertRead(alertId));
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        CDP Climate Change Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Reporting Year: {data.reporting_year}
      </Typography>

      {/* Row 1: Score + Readiness + Timeline + Gap Summary */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <ScoreCard
            predictedScore={data.predicted_score}
            predictedLevel={data.predicted_level}
            predictedBand={data.predicted_band}
            previousScore={data.previous_score}
            scoreDelta={data.score_delta}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <ReadinessScore
            readinessPct={data.readiness_pct}
            completionPct={data.completion_pct}
            answered={data.answered_questions}
            total={data.total_questions}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <TimelineCountdown
            deadline={data.submission_deadline}
            daysRemaining={data.days_until_deadline}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <GapSummary
            total={data.gap_summary.total}
            critical={data.gap_summary.critical}
            high={data.gap_summary.high}
            medium={data.gap_summary.medium}
            low={data.gap_summary.low}
            resolved={data.gap_summary.resolved}
          />
        </Grid>
      </Grid>

      {/* Row 2: Category Radar + A-List + Module Progress */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={5}>
          <CategoryRadar categories={data.category_scores} />
        </Grid>
        <Grid item xs={12} md={3}>
          <AListChecklist requirements={data.a_level_status} />
        </Grid>
        <Grid item xs={12} md={4}>
          <ModuleProgressComponent modules={data.module_progress} />
        </Grid>
      </Grid>

      {/* Row 3: Trend Chart + Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <TrendChart scores={[]} />
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alerts
              </Typography>
              {alerts.length === 0 ? (
                <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ py: 2 }}>
                  No alerts.
                </Typography>
              ) : (
                alerts
                  .filter((a) => !a.is_read)
                  .slice(0, 5)
                  .map((alert) => (
                    <Alert
                      key={alert.id}
                      severity={alert.severity}
                      onClose={() => handleAlertRead(alert.id)}
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
    </Box>
  );
};

export default DashboardPage;
