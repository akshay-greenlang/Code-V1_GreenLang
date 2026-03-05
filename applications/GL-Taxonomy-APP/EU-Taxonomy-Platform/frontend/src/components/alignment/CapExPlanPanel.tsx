/**
 * CapExPlanPanel - CapEx plan management for transitional activities.
 *
 * Displays active CapEx plans with timeline, milestones, and spend tracking.
 * Per Article 8 DA, CapEx plans add to aligned CapEx numerator for up to 5-10 years.
 */

import React from 'react';
import {
  Card, CardContent, Typography, Box, Chip, LinearProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
} from '@mui/material';
import { AccountBalance, Event, TrendingUp } from '@mui/icons-material';

interface CapExPlan {
  id: string;
  activity: string;
  objective: string;
  totalBudgetEur: number;
  spentEur: number;
  startYear: number;
  endYear: number;
  milestones: number;
  milestonesCompleted: number;
  status: string;
}

const DEMO_PLANS: CapExPlan[] = [
  { id: '1', activity: '7.2 - Building renovation', objective: 'CCM', totalBudgetEur: 12000000, spentEur: 4800000, startYear: 2023, endYear: 2028, milestones: 8, milestonesCompleted: 3, status: 'on_track' },
  { id: '2', activity: '4.1 - Electricity generation (solar)', objective: 'CCM', totalBudgetEur: 8000000, spentEur: 6400000, startYear: 2022, endYear: 2026, milestones: 6, milestonesCompleted: 5, status: 'ahead' },
  { id: '3', activity: '6.5 - Transport by motorbikes/cars', objective: 'CCM', totalBudgetEur: 3000000, spentEur: 900000, startYear: 2024, endYear: 2029, milestones: 5, milestonesCompleted: 1, status: 'at_risk' },
];

const CapExPlanPanel: React.FC = () => {
  const totalBudget = DEMO_PLANS.reduce((s, p) => s + p.totalBudgetEur, 0);
  const totalSpent = DEMO_PLANS.reduce((s, p) => s + p.spentEur, 0);
  const pctSpent = Math.round((totalSpent / totalBudget) * 100);

  const statusColor = (status: string): 'success' | 'warning' | 'error' | 'default' => {
    if (status === 'ahead') return 'success';
    if (status === 'on_track') return 'success';
    if (status === 'at_risk') return 'warning';
    if (status === 'delayed') return 'error';
    return 'default';
  };

  const fmtEur = (v: number) => new Intl.NumberFormat('en-EU', { notation: 'compact', maximumFractionDigits: 1 }).format(v);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccountBalance color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>CapEx Plans</Typography>
          </Box>
          <Chip label={`${DEMO_PLANS.length} Active Plans`} size="small" color="primary" variant="outlined" />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>Total Budget Utilization</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>{fmtEur(totalSpent)} / {fmtEur(totalBudget)} ({pctSpent}%)</Typography>
          </Box>
          <LinearProgress variant="determinate" value={pctSpent} sx={{ height: 8, borderRadius: 4 }} />
        </Box>

        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Activity</TableCell>
                <TableCell>Obj.</TableCell>
                <TableCell align="right">Budget</TableCell>
                <TableCell align="right">Spent</TableCell>
                <TableCell align="center">Timeline</TableCell>
                <TableCell align="center">Milestones</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {DEMO_PLANS.map((plan) => (
                <TableRow key={plan.id} hover>
                  <TableCell sx={{ fontWeight: 500, fontSize: '0.85rem' }}>{plan.activity}</TableCell>
                  <TableCell><Chip label={plan.objective} size="small" variant="outlined" sx={{ fontSize: '0.65rem' }} /></TableCell>
                  <TableCell align="right" sx={{ fontSize: '0.85rem' }}>{fmtEur(plan.totalBudgetEur)}</TableCell>
                  <TableCell align="right" sx={{ fontSize: '0.85rem' }}>{fmtEur(plan.spentEur)}</TableCell>
                  <TableCell align="center" sx={{ fontSize: '0.8rem' }}>{plan.startYear} - {plan.endYear}</TableCell>
                  <TableCell align="center" sx={{ fontSize: '0.85rem' }}>{plan.milestonesCompleted}/{plan.milestones}</TableCell>
                  <TableCell>
                    <Chip label={plan.status.replace(/_/g, ' ')} size="small" color={statusColor(plan.status)} sx={{ textTransform: 'capitalize', fontSize: '0.65rem' }} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default CapExPlanPanel;
