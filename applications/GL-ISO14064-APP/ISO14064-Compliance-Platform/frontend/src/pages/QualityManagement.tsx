/**
 * GL-ISO14064-APP v1.0 - Quality Management Page
 *
 * ISO 14064-1 Clause 7 quality management interface.
 * Sections:
 *   - Quality plans overview
 *   - Quality procedures list
 *   - Data quality matrix by category & dimension
 *   - Corrective actions tracker
 */

import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Tabs, Tab, Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchQualityPlans, fetchProcedures, fetchDataQualityMatrix,
  fetchCorrectiveActions,
} from '../store/slices/qualitySlice';
import type {
  QualityPlan, QualityProcedure, DataQualityEntry, CorrectiveAction,
} from '../store/slices/qualitySlice';

function statusChip(status: string) {
  const map: Record<string, 'success' | 'warning' | 'error' | 'info' | 'default'> = {
    active: 'success', approved: 'success', completed: 'success', resolved: 'success',
    draft: 'default', pending: 'warning', in_progress: 'info',
    overdue: 'error', open: 'error',
  };
  return (
    <Chip
      label={status.replace(/_/g, ' ')}
      size="small"
      color={map[status] ?? 'default'}
    />
  );
}

function priorityChip(priority: string) {
  const map: Record<string, 'error' | 'warning' | 'info' | 'default'> = {
    critical: 'error', high: 'error', medium: 'warning', low: 'info',
  };
  return <Chip label={priority} size="small" color={map[priority] ?? 'default'} variant="outlined" />;
}

function qualityScoreBar(score: number) {
  const color = score >= 80 ? 'success' : score >= 60 ? 'warning' : 'error';
  return (
    <Box display="flex" alignItems="center" gap={1} width={160}>
      <LinearProgress
        variant="determinate"
        value={score}
        color={color as any}
        sx={{ flex: 1, height: 8, borderRadius: 4 }}
      />
      <Typography variant="caption" fontWeight={600}>{score}</Typography>
    </Box>
  );
}

const QualityManagementPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { plans, procedures, dataQualityMatrix, correctiveActions, loading, error } =
    useAppSelector((s) => (s as any).quality ?? {
      plans: [], procedures: [], dataQualityMatrix: [], correctiveActions: [], loading: false, error: null,
    });

  const [tab, setTab] = useState(0);

  // On mount load demo data (org_id = "default")
  useEffect(() => {
    dispatch(fetchQualityPlans('default'));
  }, [dispatch]);

  // When plans load, fetch related data for first plan
  useEffect(() => {
    if (plans.length > 0) {
      const planId = plans[0].plan_id;
      dispatch(fetchProcedures(planId));
      dispatch(fetchCorrectiveActions(planId));
    }
  }, [dispatch, plans]);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Quality Management (Clause 7)
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        ISO 14064-1:2018 requires organizations to establish quality management procedures for
        data collection, processing, and reporting.
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 3 }}>
        <Tab label="Quality Plans" />
        <Tab label="Procedures" />
        <Tab label="Data Quality Matrix" />
        <Tab label="Corrective Actions" />
      </Tabs>

      {/* Tab 0: Quality Plans */}
      {tab === 0 && (
        <Grid container spacing={2}>
          {(plans as QualityPlan[]).length === 0 ? (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary">No quality plans defined yet.</Typography>
                  <Button variant="contained" sx={{ mt: 1 }}>Create Quality Plan</Button>
                </CardContent>
              </Card>
            </Grid>
          ) : (
            (plans as QualityPlan[]).map((plan) => (
              <Grid item xs={12} md={6} key={plan.plan_id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="subtitle1" fontWeight={600}>
                        {plan.title}
                      </Typography>
                      {statusChip(plan.status)}
                    </Box>
                    <Typography variant="body2" color="text.secondary" mt={1}>
                      {plan.description}
                    </Typography>
                    <Box display="flex" gap={2} mt={1}>
                      <Typography variant="caption">
                        Scope: {plan.scope}
                      </Typography>
                      <Typography variant="caption">
                        Responsible: {plan.responsible_person}
                      </Typography>
                      <Typography variant="caption">
                        Review: {plan.review_frequency}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))
          )}
        </Grid>
      )}

      {/* Tab 1: Procedures */}
      {tab === 1 && (
        <TableContainer component={Card}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Procedure</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Responsible</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Frequency</TableCell>
                <TableCell align="center" sx={{ fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Next Due</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(procedures as QualityProcedure[]).length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography variant="body2" color="text.secondary">No procedures defined.</Typography>
                  </TableCell>
                </TableRow>
              ) : (
                (procedures as QualityProcedure[]).map((proc) => (
                  <TableRow key={proc.procedure_id} hover>
                    <TableCell>{proc.title}</TableCell>
                    <TableCell>{proc.procedure_type.replace(/_/g, ' ')}</TableCell>
                    <TableCell>{proc.responsible}</TableCell>
                    <TableCell>{proc.frequency}</TableCell>
                    <TableCell align="center">{statusChip(proc.status)}</TableCell>
                    <TableCell>
                      {proc.next_due_at ? new Date(proc.next_due_at).toLocaleDateString() : '-'}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Tab 2: Data Quality Matrix */}
      {tab === 2 && (
        <TableContainer component={Card}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Category</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Dimension</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Score</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Tier</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Notes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(dataQualityMatrix as DataQualityEntry[]).length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No data quality assessments available. Run an inventory first.
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                (dataQualityMatrix as DataQualityEntry[]).map((entry, i) => (
                  <TableRow key={`${entry.category}-${entry.dimension}-${i}`} hover>
                    <TableCell>{entry.category.replace(/_/g, ' ')}</TableCell>
                    <TableCell>{entry.dimension.replace(/_/g, ' ')}</TableCell>
                    <TableCell>{qualityScoreBar(entry.score)}</TableCell>
                    <TableCell>
                      <Chip label={entry.tier.replace(/_/g, ' ')} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">{entry.notes}</Typography>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Tab 3: Corrective Actions */}
      {tab === 3 && (
        <TableContainer component={Card}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Action</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Root Cause</TableCell>
                <TableCell align="center" sx={{ fontWeight: 600 }}>Priority</TableCell>
                <TableCell align="center" sx={{ fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Assigned To</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Due Date</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(correctiveActions as CorrectiveAction[]).length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography variant="body2" color="text.secondary">No corrective actions recorded.</Typography>
                  </TableCell>
                </TableRow>
              ) : (
                (correctiveActions as CorrectiveAction[]).map((ca) => (
                  <TableRow key={ca.action_id} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight={500}>{ca.title}</Typography>
                      <Typography variant="caption" color="text.secondary">{ca.description}</Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">{ca.root_cause}</Typography>
                    </TableCell>
                    <TableCell align="center">{priorityChip(ca.priority)}</TableCell>
                    <TableCell align="center">{statusChip(ca.status)}</TableCell>
                    <TableCell>{ca.assigned_to}</TableCell>
                    <TableCell>
                      {ca.due_date ? new Date(ca.due_date).toLocaleDateString() : '-'}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default QualityManagementPage;
