/**
 * GapAnalysis - Maturity spider/radar chart, gap table, action plan timeline, pillar scoring.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  LinearProgress, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent,
} from '@mui/material';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts';
import { GppBad, TrackChanges, TrendingUp, CheckCircle } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';

/* ── Demo Data ────────────────────────────────────────────────── */

const MATURITY_SCORES = [
  { dimension: 'Board Oversight', current: 85, target: 95, pillar: 'Governance' },
  { dimension: 'Management Role', current: 80, target: 90, pillar: 'Governance' },
  { dimension: 'Risk Identification', current: 75, target: 90, pillar: 'Strategy' },
  { dimension: 'Opportunity Assessment', current: 65, target: 85, pillar: 'Strategy' },
  { dimension: 'Scenario Analysis', current: 55, target: 90, pillar: 'Strategy' },
  { dimension: 'Risk Process', current: 70, target: 85, pillar: 'Risk Management' },
  { dimension: 'ERM Integration', current: 40, target: 80, pillar: 'Risk Management' },
  { dimension: 'GHG Metrics', current: 90, target: 95, pillar: 'Metrics & Targets' },
  { dimension: 'Targets & KPIs', current: 72, target: 90, pillar: 'Metrics & Targets' },
  { dimension: 'Data Quality', current: 68, target: 85, pillar: 'Metrics & Targets' },
];

const PILLAR_SUMMARY = [
  { pillar: 'Governance', avgCurrent: 83, avgTarget: 93, gap: 10, color: '#1B5E20' },
  { pillar: 'Strategy', avgCurrent: 65, avgTarget: 88, gap: 23, color: '#0D47A1' },
  { pillar: 'Risk Management', avgCurrent: 55, avgTarget: 83, gap: 28, color: '#E65100' },
  { pillar: 'Metrics & Targets', avgCurrent: 77, avgTarget: 90, gap: 13, color: '#4527A0' },
];

interface GapItem {
  id: string;
  requirement: string;
  pillar: string;
  currentMaturity: number;
  targetMaturity: number;
  gap: number;
  priority: string;
  action: string;
  owner: string;
  dueDate: string;
  status: string;
  effort: string;
}

const GAP_TABLE: GapItem[] = [
  { id: '1', requirement: 'ERM Integration', pillar: 'Risk Management', currentMaturity: 40, targetMaturity: 80, gap: 40, priority: 'critical', action: 'Implement climate risk module in ERM system', owner: 'Michael Park', dueDate: '2025-06-30', status: 'in_progress', effort: 'high' },
  { id: '2', requirement: 'Scenario Analysis (2C)', pillar: 'Strategy', currentMaturity: 55, targetMaturity: 90, gap: 35, priority: 'high', action: 'Conduct full quantitative scenario analysis with financial impact', owner: 'James Mitchell', dueDate: '2025-05-15', status: 'in_progress', effort: 'high' },
  { id: '3', requirement: 'Opportunity Quantification', pillar: 'Strategy', currentMaturity: 65, targetMaturity: 85, gap: 20, priority: 'medium', action: 'Complete revenue sizing for all opportunity categories', owner: 'Sarah Chen', dueDate: '2025-07-01', status: 'not_started', effort: 'medium' },
  { id: '4', requirement: 'Data Quality Framework', pillar: 'Metrics & Targets', currentMaturity: 68, targetMaturity: 85, gap: 17, priority: 'medium', action: 'Implement automated data quality scoring and validation', owner: 'Aisha Rahman', dueDate: '2025-08-01', status: 'not_started', effort: 'medium' },
  { id: '5', requirement: 'Scope 3 Coverage', pillar: 'Metrics & Targets', currentMaturity: 72, targetMaturity: 90, gap: 18, priority: 'medium', action: 'Expand Scope 3 to cover all 15 GHG Protocol categories', owner: 'Aisha Rahman', dueDate: '2025-09-01', status: 'planning', effort: 'high' },
  { id: '6', requirement: 'Risk Process Documentation', pillar: 'Risk Management', currentMaturity: 70, targetMaturity: 85, gap: 15, priority: 'low', action: 'Document risk identification and assessment methodology', owner: 'Michael Park', dueDate: '2025-10-01', status: 'not_started', effort: 'low' },
  { id: '7', requirement: 'Board Climate Competency', pillar: 'Governance', currentMaturity: 85, targetMaturity: 95, gap: 10, priority: 'low', action: 'Annual board climate training and competency assessment', owner: 'Legal/Governance', dueDate: '2025-12-01', status: 'completed', effort: 'low' },
];

const MATURITY_TREND = [
  { quarter: 'Q1 2024', governance: 70, strategy: 50, riskMgmt: 35, metrics: 60 },
  { quarter: 'Q2 2024', governance: 75, strategy: 55, riskMgmt: 40, metrics: 65 },
  { quarter: 'Q3 2024', governance: 78, strategy: 58, riskMgmt: 45, metrics: 70 },
  { quarter: 'Q4 2024', governance: 80, strategy: 62, riskMgmt: 50, metrics: 74 },
  { quarter: 'Q1 2025', governance: 83, strategy: 65, riskMgmt: 55, metrics: 77 },
];

/* ── Component ─────────────────────────────────────────────────── */

const GapAnalysis: React.FC = () => {
  const [priorityFilter, setPriorityFilter] = useState('all');

  const filteredGaps = priorityFilter === 'all'
    ? GAP_TABLE
    : GAP_TABLE.filter((g) => g.priority === priorityFilter);

  const overallCurrent = Math.round(MATURITY_SCORES.reduce((s, m) => s + m.current, 0) / MATURITY_SCORES.length);
  const overallTarget = Math.round(MATURITY_SCORES.reduce((s, m) => s + m.target, 0) / MATURITY_SCORES.length);
  const criticalGaps = GAP_TABLE.filter((g) => g.priority === 'critical' || g.priority === 'high').length;
  const actionsCompleted = GAP_TABLE.filter((g) => g.status === 'completed').length;

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
          TCFD maturity assessment, gap identification, and remediation action planning
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Overall Maturity"
            value={overallCurrent}
            format="percent"
            icon={<TrackChanges />}
            subtitle={`Target: ${overallTarget}%`}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Average Gap"
            value={overallTarget - overallCurrent}
            icon={<GppBad />}
            subtitle="points to target"
            color="error"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Critical/High Gaps"
            value={criticalGaps}
            icon={<TrendingUp />}
            subtitle={`of ${GAP_TABLE.length} total`}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Actions Completed"
            value={actionsCompleted}
            icon={<CheckCircle />}
            subtitle={`of ${GAP_TABLE.length} total`}
            color="success"
          />
        </Grid>
      </Grid>

      {/* Radar Chart + Pillar Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Maturity Spider Chart</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={MATURITY_SCORES}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" fontSize={10} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Current" dataKey="current" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                  <Radar name="Target" dataKey="target" stroke="#C62828" fill="#C62828" fillOpacity={0.1} strokeWidth={2} strokeDasharray="5 5" />
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
              <Typography variant="h6" sx={{ mb: 2 }}>Pillar Gap Summary</Typography>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={PILLAR_SUMMARY} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="pillar" width={120} fontSize={11} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Bar dataKey="avgCurrent" name="Current" barSize={12}>
                    {PILLAR_SUMMARY.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Bar>
                  <Bar dataKey="avgTarget" name="Target" fill="#E0E0E0" barSize={12} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Pillar Breakdown</Typography>
              {PILLAR_SUMMARY.map((pillar) => (
                <Box key={pillar.pillar} sx={{ mb: 1.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{pillar.pillar}</Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {pillar.avgCurrent}% / {pillar.avgTarget}% (gap: {pillar.gap}pts)
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={pillar.avgCurrent}
                    sx={{
                      height: 8, borderRadius: 4,
                      backgroundColor: '#E0E0E0',
                      '& .MuiLinearProgress-bar': { backgroundColor: pillar.color },
                    }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Gap Table */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Gap Remediation Actions</Typography>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Priority</InputLabel>
              <Select
                value={priorityFilter}
                label="Priority"
                onChange={(e: SelectChangeEvent) => setPriorityFilter(e.target.value)}
              >
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
                  <TableCell>Requirement</TableCell>
                  <TableCell>Pillar</TableCell>
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
                    <TableCell sx={{ fontWeight: 500 }}>{gap.requirement}</TableCell>
                    <TableCell>
                      <Chip label={gap.pillar} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, justifyContent: 'center' }}>
                        <Typography variant="body2" sx={{ fontWeight: 600, color: gap.gap > 25 ? 'error.main' : gap.gap > 15 ? 'warning.main' : 'text.primary' }}>
                          {gap.gap}pts
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={gap.priority}
                        size="small"
                        color={getPriorityColor(gap.priority)}
                        sx={{ textTransform: 'capitalize' }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.8rem', maxWidth: 200 }}>{gap.action}</TableCell>
                    <TableCell sx={{ fontSize: '0.8rem' }}>{gap.owner}</TableCell>
                    <TableCell align="center" sx={{ fontSize: '0.8rem' }}>
                      {new Date(gap.dueDate).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={gap.status.replace(/_/g, ' ')}
                        size="small"
                        color={getStatusColor(gap.status)}
                        sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={gap.effort}
                        size="small"
                        variant="outlined"
                        sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Maturity Trend */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Maturity Trend Over Time</Typography>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={MATURITY_TREND}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="quarter" fontSize={11} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v: number) => [`${v}%`, '']} />
              <Legend />
              <Bar dataKey="governance" name="Governance" fill="#1B5E20" />
              <Bar dataKey="strategy" name="Strategy" fill="#0D47A1" />
              <Bar dataKey="riskMgmt" name="Risk Mgmt" fill="#E65100" />
              <Bar dataKey="metrics" name="Metrics" fill="#4527A0" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default GapAnalysis;
