/**
 * ValidationChecker - SBTi criteria validation, readiness assessment, and issue tracking.
 *
 * Shows criteria checklist, coverage gauge, ambition indicator, readiness bar, and issue list.
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Button, Alert, Chip,
  FormControl, InputLabel, Select, MenuItem, SelectChangeEvent,
  LinearProgress,
} from '@mui/material';
import { PlayArrow, Refresh, CheckCircle, Warning, Error as ErrorIcon } from '@mui/icons-material';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts';
import CriteriaChecklist from '../components/validation/CriteriaChecklist';
import CoverageGauge from '../components/validation/CoverageGauge';
import AmbitionIndicator from '../components/validation/AmbitionIndicator';
import ReadinessBar from '../components/validation/ReadinessBar';
import IssueList from '../components/validation/IssueList';
import ScoreGauge from '../components/common/ScoreGauge';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { runValidation, fetchChecklist, fetchReadiness, selectValidationResult, selectChecklist, selectReadiness, selectValidationLoading, selectValidating } from '../store/slices/validationSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

/* Demo data */
const DEMO_CHECKLIST = [
  { criterion_code: 'C1', criterion_name: 'Base Year Recency', category: 'Base Year', status: 'pass' },
  { criterion_code: 'C2', criterion_name: 'S1+S2 Coverage >= 95%', category: 'Coverage', status: 'pass' },
  { criterion_code: 'C3', criterion_name: 'S3 Coverage >= 67%', category: 'Coverage', status: 'warning' },
  { criterion_code: 'C4', criterion_name: 'Near-term 5-10yr Timeframe', category: 'Timeframe', status: 'pass' },
  { criterion_code: 'C5', criterion_name: 'Min 4.2% Annual Reduction (1.5C)', category: 'Ambition', status: 'pass' },
  { criterion_code: 'C6', criterion_name: 'No Offsets in Target Boundary', category: 'Boundary', status: 'pass' },
  { criterion_code: 'C7', criterion_name: 'Long-term Target by 2050', category: 'Long-term', status: 'pass' },
  { criterion_code: 'C8', criterion_name: 'FLAG Sector Assessment', category: 'FLAG', status: 'not_applicable' },
  { criterion_code: 'C9', criterion_name: 'Scope 3 Screening Complete', category: 'Scope 3', status: 'pass' },
  { criterion_code: 'C10', criterion_name: '5-Year Review Scheduled', category: 'Review', status: 'warning' },
  { criterion_code: 'C11', criterion_name: 'Net-Zero Minimum 90% Reduction', category: 'Long-term', status: 'pass' },
  { criterion_code: 'C12', criterion_name: 'Bioenergy Accounting', category: 'Special', status: 'not_applicable' },
];

const DEMO_READINESS = { readiness_score: 78, category_scores: [
  { category: 'Base Year', score: 95 },
  { category: 'Coverage', score: 72 },
  { category: 'Ambition', score: 88 },
  { category: 'Timeframe', score: 90 },
  { category: 'Boundary', score: 85 },
  { category: 'Scope 3', score: 65 },
], blockers: ['Scope 3 coverage below 67% threshold', '5-Year review schedule not confirmed'] };

const DEMO_ISSUES = [
  { id: '1', criterion_code: 'C3', description: 'Scope 3 coverage is 64%, below the 67% minimum threshold', severity: 'high', recommendation: 'Include Category 11 (Use of Sold Products) to reach 68%', status: 'open' },
  { id: '2', criterion_code: 'C10', description: 'Five-year review date not formally scheduled with SBTi', severity: 'medium', recommendation: 'Contact SBTi to schedule review for 2029', status: 'open' },
  { id: '3', criterion_code: 'C5', description: 'Annual reduction rate is 4.1% - marginally below 4.2% minimum', severity: 'low', recommendation: 'Minor pathway adjustment needed to meet 1.5C threshold', status: 'resolved' },
];

const ValidationChecker: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const result = useAppSelector(selectValidationResult);
  const checklist = useAppSelector(selectChecklist);
  const readiness = useAppSelector(selectReadiness);
  const loading = useAppSelector(selectValidationLoading);
  const validating = useAppSelector(selectValidating);

  useEffect(() => {
    dispatch(fetchChecklist(orgId));
    dispatch(fetchReadiness(orgId));
  }, [dispatch, orgId]);

  const checklistData = checklist.length > 0 ? checklist : DEMO_CHECKLIST;
  const readinessData = readiness || DEMO_READINESS;

  const passCount = checklistData.filter((c) => c.status === 'pass').length;
  const warnCount = checklistData.filter((c) => c.status === 'warning').length;
  const failCount = checklistData.filter((c) => c.status === 'fail').length;
  const applicableCount = checklistData.filter((c) => c.status !== 'not_applicable').length;

  const radarData = readinessData.category_scores.map((cs) => ({
    category: cs.category,
    score: cs.score,
    target: 80,
  }));

  const handleRunValidation = () => {
    dispatch(runValidation('tgt_1'));
  };

  if (loading && checklist.length === 0) return <LoadingSpinner message="Loading validation..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Validation Checker</Typography>
          <Typography variant="body2" color="text.secondary">
            Assess SBTi criteria compliance and submission readiness
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={validating ? <Refresh /> : <PlayArrow />}
          onClick={handleRunValidation}
          disabled={validating}
        >
          {validating ? 'Validating...' : 'Run Validation'}
        </Button>
      </Box>

      {/* KPI Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>Readiness Score</Typography>
              <ScoreGauge value={readinessData.readiness_score} size={90} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CheckCircle color="success" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{passCount}</Typography>
              <Typography variant="body2" color="text.secondary">Criteria Passed</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Warning color="warning" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{warnCount}</Typography>
              <Typography variant="body2" color="text.secondary">Warnings</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <ErrorIcon color="error" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{failCount}</Typography>
              <Typography variant="body2" color="text.secondary">Failures</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Radar + Checklist */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Category Readiness</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="category" fontSize={10} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Score" dataKey="score" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                  <Radar name="Target" dataKey="target" stroke="#C62828" fill="none" strokeWidth={1.5} strokeDasharray="5 5" />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          <CriteriaChecklist checklist={checklistData as any} />
        </Grid>
      </Grid>

      {/* Blockers & Issues */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Blockers</Typography>
              {readinessData.blockers.length === 0 ? (
                <Alert severity="success">No blockers identified - ready for submission!</Alert>
              ) : (
                readinessData.blockers.map((blocker, idx) => (
                  <Alert key={idx} severity="warning" sx={{ mb: 1 }}>
                    {blocker}
                  </Alert>
                ))
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <IssueList issues={DEMO_ISSUES as any} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default ValidationChecker;
