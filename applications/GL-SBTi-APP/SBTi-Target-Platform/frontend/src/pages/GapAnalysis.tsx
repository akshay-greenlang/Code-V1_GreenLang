/**
 * GapAnalysis - SBTi readiness gap assessment, action planning, and remediation tracking.
 *
 * Displays maturity spider chart, gap table with priorities, action timeline,
 * and category scoring.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  LinearProgress, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent,
  Alert, Button,
} from '@mui/material';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts';
import { GppBad, TrackChanges, TrendingUp, CheckCircle, PlayArrow } from '@mui/icons-material';
import ScoreGauge from '../components/common/ScoreGauge';

/* Demo Data */
const MATURITY_SCORES = [
  { dimension: 'Base Year Inventory', current: 95, target: 100, category: 'Data Foundation' },
  { dimension: 'Scope 1+2 Coverage', current: 98, target: 100, category: 'Coverage' },
  { dimension: 'Scope 3 Coverage', current: 68, target: 80, category: 'Coverage' },
  { dimension: 'Target Ambition', current: 88, target: 95, category: 'Ambition' },
  { dimension: 'Pathway Calculation', current: 82, target: 90, category: 'Methodology' },
  { dimension: 'Validation Readiness', current: 78, target: 90, category: 'Submission' },
  { dimension: 'Progress Tracking', current: 75, target: 85, category: 'Monitoring' },
  { dimension: 'Temperature Scoring', current: 70, target: 85, category: 'Analytics' },
  { dimension: 'Framework Alignment', current: 65, target: 80, category: 'Compliance' },
  { dimension: 'Review Preparation', current: 33, target: 80, category: 'Lifecycle' },
];

const CATEGORY_SUMMARY = [
  { category: 'Data Foundation', avgCurrent: 95, avgTarget: 100, gap: 5, color: '#1B5E20' },
  { category: 'Coverage', avgCurrent: 83, avgTarget: 90, gap: 7, color: '#0D47A1' },
  { category: 'Ambition', avgCurrent: 88, avgTarget: 95, gap: 7, color: '#7B1FA2' },
  { category: 'Methodology', avgCurrent: 82, avgTarget: 90, gap: 8, color: '#E65100' },
  { category: 'Submission', avgCurrent: 78, avgTarget: 90, gap: 12, color: '#C62828' },
  { category: 'Monitoring', avgCurrent: 75, avgTarget: 85, gap: 10, color: '#006064' },
  { category: 'Analytics', avgCurrent: 70, avgTarget: 85, gap: 15, color: '#33691E' },
  { category: 'Compliance', avgCurrent: 65, avgTarget: 80, gap: 15, color: '#880E4F' },
  { category: 'Lifecycle', avgCurrent: 33, avgTarget: 80, gap: 47, color: '#4527A0' },
];

interface GapAction {
  id: string;
  dimension: string;
  category: string;
  gap: number;
  priority: string;
  action: string;
  owner: string;
  dueDate: string;
  status: string;
  effort: string;
}

const GAP_ACTIONS: GapAction[] = [
  { id: '1', dimension: 'Review Preparation', category: 'Lifecycle', gap: 47, priority: 'critical', action: 'Begin 5-year review preparation checklist', owner: 'Sustainability Lead', dueDate: '2025-06-30', status: 'not_started', effort: 'high' },
  { id: '2', dimension: 'Framework Alignment', category: 'Compliance', gap: 15, priority: 'high', action: 'Map SBTi requirements to SEC and ISSB frameworks', owner: 'Compliance Team', dueDate: '2025-09-01', status: 'in_progress', effort: 'medium' },
  { id: '3', dimension: 'Temperature Scoring', category: 'Analytics', gap: 15, priority: 'high', action: 'Implement automated temperature scoring calculation', owner: 'Data Science', dueDate: '2025-07-15', status: 'in_progress', effort: 'high' },
  { id: '4', dimension: 'Scope 3 Coverage', category: 'Coverage', gap: 12, priority: 'high', action: 'Expand Scope 3 to include Category 11', owner: 'Carbon Accounting', dueDate: '2025-05-30', status: 'planning', effort: 'medium' },
  { id: '5', dimension: 'Validation Readiness', category: 'Submission', gap: 12, priority: 'medium', action: 'Complete all validation criteria checklist items', owner: 'Sustainability Lead', dueDate: '2025-08-01', status: 'in_progress', effort: 'medium' },
  { id: '6', dimension: 'Progress Tracking', category: 'Monitoring', gap: 10, priority: 'medium', action: 'Automate annual progress data collection pipeline', owner: 'Data Engineering', dueDate: '2025-10-01', status: 'not_started', effort: 'high' },
  { id: '7', dimension: 'Pathway Calculation', category: 'Methodology', gap: 8, priority: 'low', action: 'Validate SDA pathway parameters against latest sector benchmarks', owner: 'Carbon Accounting', dueDate: '2025-12-01', status: 'not_started', effort: 'low' },
  { id: '8', dimension: 'Target Ambition', category: 'Ambition', gap: 7, priority: 'low', action: 'Review target ambition level against updated 1.5C requirements', owner: 'Sustainability Lead', dueDate: '2025-12-01', status: 'completed', effort: 'low' },
];

const GapAnalysis: React.FC = () => {
  const [priorityFilter, setPriorityFilter] = useState('all');

  const filteredGaps = priorityFilter === 'all'
    ? GAP_ACTIONS
    : GAP_ACTIONS.filter((g) => g.priority === priorityFilter);

  const overallCurrent = Math.round(MATURITY_SCORES.reduce((s, m) => s + m.current, 0) / MATURITY_SCORES.length);
  const overallTarget = Math.round(MATURITY_SCORES.reduce((s, m) => s + m.target, 0) / MATURITY_SCORES.length);
  const criticalGaps = GAP_ACTIONS.filter((g) => g.priority === 'critical' || g.priority === 'high').length;
  const actionsCompleted = GAP_ACTIONS.filter((g) => g.status === 'completed').length;

  const getPriorityColor = (priority: string): 'error' | 'warning' | 'info' | 'default' => {
    if (priority === 'critical') return 'error';
    if (priority === 'high') return 'warning';
    if (priority === 'medium') return 'info';
    return 'default';
  };

  const getStatusColor = (status: string): 'success' | 'warning' | 'info' | 'default' | 'error' => {
    if (status === 'completed') return 'success';
    if (status === 'in_progress') return 'warning';
    if (status === 'planning') return 'info';
    if (status === 'not_started') return 'default';
    return 'error';
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Gap Analysis</Typography>
        <Typography variant="body2" color="text.secondary">
          SBTi readiness maturity assessment, gap identification, and remediation action planning
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 0.5 }}>Overall Maturity</Typography>
              <ScoreGauge value={overallCurrent} size={80} />
              <Typography variant="caption" color="text.secondary">Target: {overallTarget}%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <GppBad color="error" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{overallTarget - overallCurrent}pts</Typography>
              <Typography variant="body2" color="text.secondary">Average Gap</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUp color="warning" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{criticalGaps}</Typography>
              <Typography variant="body2" color="text.secondary">Critical/High Gaps</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CheckCircle color="success" sx={{ fontSize: 32, mb: 0.5 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{actionsCompleted}/{GAP_ACTIONS.length}</Typography>
              <Typography variant="body2" color="text.secondary">Actions Completed</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Radar + Category Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Maturity Spider Chart</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={MATURITY_SCORES}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" fontSize={9} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Current" dataKey="current" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                  <Radar name="Target" dataKey="target" stroke="#C62828" fill="none" strokeWidth={2} strokeDasharray="5 5" />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Category Gap Summary</Typography>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={CATEGORY_SUMMARY} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="category" width={110} fontSize={10} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Bar dataKey="avgCurrent" name="Current" barSize={10}>
                    {CATEGORY_SUMMARY.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}
                  </Bar>
                  <Bar dataKey="avgTarget" name="Target" fill="#E0E0E0" barSize={10} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Category Breakdown</Typography>
              {CATEGORY_SUMMARY.map((cat) => (
                <Box key={cat.category} sx={{ mb: 1.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{cat.category}</Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {cat.avgCurrent}% / {cat.avgTarget}% (gap: {cat.gap}pts)
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={cat.avgCurrent}
                    sx={{ height: 6, borderRadius: 3, backgroundColor: '#E0E0E0', '& .MuiLinearProgress-bar': { backgroundColor: cat.color } }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Gap Action Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Gap Remediation Actions</Typography>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Priority</InputLabel>
              <Select value={priorityFilter} label="Priority" onChange={(e: SelectChangeEvent) => setPriorityFilter(e.target.value)}>
                <MenuItem value="all">All Priorities</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Dimension</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell align="center">Gap</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>Action</TableCell>
                  <TableCell>Owner</TableCell>
                  <TableCell align="center">Due</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Effort</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredGaps.sort((a, b) => b.gap - a.gap).map((gap) => (
                  <TableRow key={gap.id} hover>
                    <TableCell sx={{ fontWeight: 500, fontSize: '0.85rem' }}>{gap.dimension}</TableCell>
                    <TableCell>
                      <Chip label={gap.category} size="small" variant="outlined" sx={{ fontSize: '0.65rem' }} />
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2" sx={{ fontWeight: 600, color: gap.gap > 20 ? 'error.main' : gap.gap > 10 ? 'warning.main' : 'text.primary' }}>
                        {gap.gap}pts
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip label={gap.priority} size="small" color={getPriorityColor(gap.priority)} sx={{ textTransform: 'capitalize' }} />
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.8rem', maxWidth: 220 }}>{gap.action}</TableCell>
                    <TableCell sx={{ fontSize: '0.8rem' }}>{gap.owner}</TableCell>
                    <TableCell align="center" sx={{ fontSize: '0.8rem' }}>
                      {new Date(gap.dueDate).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Chip label={gap.status.replace(/_/g, ' ')} size="small" color={getStatusColor(gap.status)} sx={{ textTransform: 'capitalize', fontSize: '0.65rem' }} />
                    </TableCell>
                    <TableCell>
                      <Chip label={gap.effort} size="small" variant="outlined" sx={{ textTransform: 'capitalize', fontSize: '0.65rem' }} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default GapAnalysis;
