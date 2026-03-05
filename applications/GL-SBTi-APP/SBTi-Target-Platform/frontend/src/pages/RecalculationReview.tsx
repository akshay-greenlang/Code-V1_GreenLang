/**
 * RecalculationReview - Base year recalculation triggers and 5-year review management.
 *
 * Monitors structural changes, threshold checks, recalculation requests,
 * and the five-year review lifecycle.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Button, Alert, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Stepper, Step, StepLabel, StepContent, LinearProgress,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Refresh, Warning, CheckCircle, Schedule, Flag } from '@mui/icons-material';
import ChangeMonitor from '../components/recalculation/ChangeMonitor';
import TriggerAlerts from '../components/recalculation/TriggerAlerts';
import ReviewTimeline from '../components/recalculation/ReviewTimeline';
import ReviewChecklist from '../components/recalculation/ReviewChecklist';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchThresholdChecks, fetchRecalculations,
  selectThresholdChecks, selectRecalculations, selectRecalculationLoading,
} from '../store/slices/recalculationSlice';
import {
  fetchReview, fetchReviewReadiness,
  selectCurrentReview, selectReviewReadiness, selectReviewLoading,
} from '../store/slices/reviewSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_THRESHOLDS = [
  { id: '1', change_type: 'Acquisition', description: 'Acquired TechSub Ltd (2,500 tCO2e)', impact_pct: 2.8, threshold_pct: 5, triggered: false, date: '2025-01-15' },
  { id: '2', change_type: 'Methodology Change', description: 'Updated Scope 2 to market-based method', impact_pct: 6.2, threshold_pct: 5, triggered: true, date: '2024-11-20' },
  { id: '3', change_type: 'Divestiture', description: 'Sold manufacturing division', impact_pct: -12.5, threshold_pct: 5, triggered: true, date: '2024-09-01' },
  { id: '4', change_type: 'Error Correction', description: 'Corrected Scope 3 Cat 1 calculation', impact_pct: 3.1, threshold_pct: 5, triggered: false, date: '2025-02-10' },
  { id: '5', change_type: 'Outsourcing', description: 'Outsourced logistics to 3PL provider', impact_pct: -4.8, threshold_pct: 5, triggered: false, date: '2025-02-01' },
];

const DEMO_RECALCULATIONS = [
  { id: 'rc_1', trigger_reason: 'Divestiture of manufacturing division', status: 'completed' as const, original_base: 300000, recalculated_base: 262500, impact_pct: -12.5, requested_date: '2024-09-15', completed_date: '2024-10-30', approved_by: 'Internal Review Board' },
  { id: 'rc_2', trigger_reason: 'Scope 2 methodology change to market-based', status: 'pending' as const, original_base: 262500, recalculated_base: 246150, impact_pct: -6.2, requested_date: '2025-01-10', completed_date: null, approved_by: null },
];

const DEMO_REVIEW = {
  id: 'rv_1',
  organization_id: 'org_default',
  review_cycle: 1,
  review_date: '2029-06-15',
  status: 'not_started' as const,
  days_remaining: 1563,
  checklist: [
    { item: 'Updated emissions inventory', completed: true },
    { item: 'Recalculated base year if needed', completed: true },
    { item: 'Reviewed target ambition level', completed: false },
    { item: 'Assessed sector pathway updates', completed: false },
    { item: 'Updated Scope 3 screening', completed: false },
    { item: 'Prepared submission documentation', completed: false },
  ],
};

const DEMO_READINESS = {
  readiness_pct: 33,
  items_completed: 2,
  items_total: 6,
  blockers: ['Target ambition level not yet reviewed', 'Scope 3 screening needs update'],
};

const RecalculationReview: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const thresholdChecks = useAppSelector(selectThresholdChecks);
  const recalculations = useAppSelector(selectRecalculations);
  const review = useAppSelector(selectCurrentReview);
  const reviewReadiness = useAppSelector(selectReviewReadiness);
  const recalcLoading = useAppSelector(selectRecalculationLoading);
  const reviewLoading = useAppSelector(selectReviewLoading);

  useEffect(() => {
    dispatch(fetchThresholdChecks(orgId));
    dispatch(fetchRecalculations(orgId));
    dispatch(fetchReview(orgId));
    dispatch(fetchReviewReadiness(orgId));
  }, [dispatch, orgId]);

  const thresholds = thresholdChecks.length > 0 ? thresholdChecks : DEMO_THRESHOLDS;
  const recalcs = recalculations.length > 0 ? recalculations : DEMO_RECALCULATIONS;
  const reviewData = review || DEMO_REVIEW;
  const readiness = reviewReadiness || DEMO_READINESS;

  const triggeredCount = thresholds.filter((t: any) => t.triggered).length;

  if (recalcLoading && thresholdChecks.length === 0) return <LoadingSpinner message="Loading recalculation data..." />;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Recalculation & Review</Typography>
        <Typography variant="body2" color="text.secondary">
          Base year recalculation triggers and five-year target review management
        </Typography>
      </Box>

      {triggeredCount > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {triggeredCount} structural change(s) exceed the 5% significance threshold and may require base year recalculation.
        </Alert>
      )}

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Warning color="warning" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{triggeredCount}</Typography>
              <Typography variant="body2" color="text.secondary">Triggered Changes</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Refresh color="primary" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{recalcs.length}</Typography>
              <Typography variant="body2" color="text.secondary">Recalculations</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Schedule color="primary" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{reviewData.days_remaining}</Typography>
              <Typography variant="body2" color="text.secondary">Days to Next Review</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Flag color="primary" sx={{ fontSize: 32 }} />
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{readiness.readiness_pct}%</Typography>
              <Typography variant="body2" color="text.secondary">Review Readiness</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Threshold Checks */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Structural Change Monitor (5% Significance Threshold)</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Change Type</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell align="center">Impact</TableCell>
                  <TableCell align="center">Threshold</TableCell>
                  <TableCell align="center">Triggered</TableCell>
                  <TableCell>Date</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {thresholds.map((t: any) => (
                  <TableRow key={t.id} hover sx={{ backgroundColor: t.triggered ? '#FFF3E0' : 'inherit' }}>
                    <TableCell sx={{ fontWeight: 500 }}>{t.change_type}</TableCell>
                    <TableCell sx={{ fontSize: '0.85rem' }}>{t.description}</TableCell>
                    <TableCell align="center">
                      <Typography variant="body2" sx={{ fontWeight: 600, color: Math.abs(t.impact_pct) >= t.threshold_pct ? 'error.main' : 'text.primary' }}>
                        {t.impact_pct > 0 ? '+' : ''}{t.impact_pct.toFixed(1)}%
                      </Typography>
                    </TableCell>
                    <TableCell align="center">{t.threshold_pct}%</TableCell>
                    <TableCell align="center">
                      {t.triggered ? <Warning color="warning" fontSize="small" /> : <CheckCircle color="success" fontSize="small" />}
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.85rem' }}>{t.date}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Recalculation History + Review Readiness */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Recalculation History</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Reason</TableCell>
                      <TableCell align="center">Status</TableCell>
                      <TableCell align="right">Original Base</TableCell>
                      <TableCell align="right">Recalculated</TableCell>
                      <TableCell align="center">Impact</TableCell>
                      <TableCell>Requested</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {recalcs.map((r: any) => (
                      <TableRow key={r.id} hover>
                        <TableCell sx={{ fontSize: '0.85rem' }}>{r.trigger_reason}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={r.status}
                            size="small"
                            color={r.status === 'completed' ? 'success' : r.status === 'pending' ? 'warning' : 'default'}
                            sx={{ textTransform: 'capitalize' }}
                          />
                        </TableCell>
                        <TableCell align="right">{r.original_base.toLocaleString()}</TableCell>
                        <TableCell align="right">{r.recalculated_base.toLocaleString()}</TableCell>
                        <TableCell align="center" sx={{ fontWeight: 600 }}>{r.impact_pct}%</TableCell>
                        <TableCell sx={{ fontSize: '0.85rem' }}>{r.requested_date}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>5-Year Review Readiness</Typography>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>Progress</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{readiness.items_completed}/{readiness.items_total}</Typography>
                </Box>
                <LinearProgress variant="determinate" value={readiness.readiness_pct} sx={{ height: 8, borderRadius: 4 }} />
              </Box>
              {reviewData.checklist.map((item: any, idx: number) => (
                <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.5, borderBottom: '1px solid #F0F0F0' }}>
                  {item.completed ? <CheckCircle color="success" fontSize="small" /> : <Box sx={{ width: 20, height: 20, borderRadius: '50%', border: '2px solid #BDBDBD' }} />}
                  <Typography variant="body2" sx={{ textDecoration: item.completed ? 'line-through' : 'none', color: item.completed ? 'text.secondary' : 'text.primary' }}>
                    {item.item}
                  </Typography>
                </Box>
              ))}
              {readiness.blockers.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>Blockers</Typography>
                  {readiness.blockers.map((b, idx) => (
                    <Alert key={idx} severity="warning" sx={{ mb: 0.5, py: 0 }}>{b}</Alert>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RecalculationReview;
