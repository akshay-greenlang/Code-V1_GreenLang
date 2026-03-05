/**
 * Governance - TCFD Pillar 1: Board oversight and management climate governance.
 *
 * Shows governance structure, committee tracker, management roles matrix,
 * maturity radar chart (10 dimensions), and disclosure panel.
 */

import React, { useEffect, useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, LinearProgress, Button, Alert } from '@mui/material';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { AccountBalance, People, School, Star, Refresh } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';
import LoadingSpinner from '../components/common/LoadingSpinner';

const Governance: React.FC = () => {
  // Demo data
  const maturityData = useMemo(() => [
    { dimension: 'Board Oversight', score: 85, fullMark: 100 },
    { dimension: 'Committee Structure', score: 90, fullMark: 100 },
    { dimension: 'Management Roles', score: 75, fullMark: 100 },
    { dimension: 'Climate Competency', score: 60, fullMark: 100 },
    { dimension: 'Strategic Integration', score: 80, fullMark: 100 },
    { dimension: 'Risk Integration', score: 70, fullMark: 100 },
    { dimension: 'Remuneration Linkage', score: 55, fullMark: 100 },
    { dimension: 'Reporting Frequency', score: 90, fullMark: 100 },
    { dimension: 'Training Programs', score: 45, fullMark: 100 },
    { dimension: 'Stakeholder Engagement', score: 65, fullMark: 100 },
  ], []);

  const committees = useMemo(() => [
    { name: 'Sustainability Committee', type: 'Board Committee', chair: 'Dr. Elena Torres', meetings: 'Quarterly', climateItems: 8, totalItems: 12, nextMeeting: '2025-04-15' },
    { name: 'Risk Committee', type: 'Board Committee', chair: 'James Mitchell', meetings: 'Monthly', climateItems: 3, totalItems: 15, nextMeeting: '2025-03-20' },
    { name: 'Climate Working Group', type: 'Management', chair: 'Sarah Chen', meetings: 'Bi-weekly', climateItems: 12, totalItems: 12, nextMeeting: '2025-03-10' },
  ], []);

  const roles = useMemo(() => [
    { name: 'Dr. Elena Torres', title: 'Chair, Sustainability Committee', type: 'Board', expertise: true, responsibilities: 'Climate strategy oversight, TCFD approval', frequency: 'Quarterly' },
    { name: 'James Mitchell', title: 'Chair, Risk Committee', type: 'Board', expertise: false, responsibilities: 'Climate risk integration, ERM oversight', frequency: 'Monthly' },
    { name: 'Sarah Chen', title: 'Chief Sustainability Officer', type: 'Executive', expertise: true, responsibilities: 'Climate program execution, target setting', frequency: 'Weekly' },
    { name: 'Michael Park', title: 'VP Risk Management', type: 'Management', expertise: true, responsibilities: 'Physical/transition risk assessment', frequency: 'Bi-weekly' },
    { name: 'Aisha Rahman', title: 'Head of ESG Reporting', type: 'Management', expertise: true, responsibilities: 'TCFD disclosure, data quality', frequency: 'Weekly' },
  ], []);

  const competencyScores = useMemo(() => [
    { area: 'Climate Science', boardMembers: 2, totalBoard: 8 },
    { area: 'GHG Accounting', boardMembers: 3, totalBoard: 8 },
    { area: 'Scenario Analysis', boardMembers: 1, totalBoard: 8 },
    { area: 'Risk Management', boardMembers: 5, totalBoard: 8 },
    { area: 'Regulatory Knowledge', boardMembers: 4, totalBoard: 8 },
    { area: 'Financial Impact', boardMembers: 6, totalBoard: 8 },
  ], []);

  const overallScore = 71.5;
  const maturityLevel = 'Managed (Level 4)';

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Governance</Typography>
          <Typography variant="body2" color="text.secondary">
            TCFD Pillar 1 -- Board oversight and management climate governance
          </Typography>
        </Box>
        <Button variant="outlined" startIcon={<Refresh />}>Run Assessment</Button>
      </Box>

      {/* KPI Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Overall Score" value={overallScore} format="percent" icon={<Star />} color="primary" subtitle={maturityLevel} />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Board Members" value={8} icon={<AccountBalance />} subtitle="3 with climate expertise" color="secondary" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Management Roles" value={roles.filter((r) => r.type !== 'Board').length} icon={<People />} subtitle="With climate accountability" color="success" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Training Coverage" value={45} format="percent" icon={<School />} trend={10} trendLabel="vs last year" color="warning" />
        </Grid>
      </Grid>

      {/* Maturity Radar + Competency */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Governance Maturity (10 Dimensions)</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={maturityData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" fontSize={10} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar name="Score" dataKey="score" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Board Climate Competency</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={competencyScores} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 8]} />
                  <YAxis type="category" dataKey="area" fontSize={11} width={120} />
                  <Tooltip />
                  <Bar dataKey="boardMembers" fill="#1B5E20" name="Members with expertise" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Committee Tracker */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Committee Tracker</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Committee</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Chair</TableCell>
                  <TableCell>Frequency</TableCell>
                  <TableCell align="center">Climate Agenda Items</TableCell>
                  <TableCell>Next Meeting</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {committees.map((c) => (
                  <TableRow key={c.name} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{c.name}</TableCell>
                    <TableCell><Chip label={c.type} size="small" variant="outlined" /></TableCell>
                    <TableCell>{c.chair}</TableCell>
                    <TableCell>{c.meetings}</TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                        <Typography variant="body2">{c.climateItems}/{c.totalItems}</Typography>
                        <LinearProgress
                          variant="determinate"
                          value={(c.climateItems / c.totalItems) * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>{new Date(c.nextMeeting).toLocaleDateString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Management Roles Matrix */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Management Roles Matrix</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Title</TableCell>
                  <TableCell>Level</TableCell>
                  <TableCell>Climate Expertise</TableCell>
                  <TableCell>Responsibilities</TableCell>
                  <TableCell>Review Frequency</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {roles.map((role) => (
                  <TableRow key={role.name} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{role.name}</TableCell>
                    <TableCell>{role.title}</TableCell>
                    <TableCell>
                      <Chip
                        label={role.type}
                        size="small"
                        color={role.type === 'Board' ? 'primary' : role.type === 'Executive' ? 'secondary' : 'default'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={role.expertise ? 'Yes' : 'No'}
                        size="small"
                        color={role.expertise ? 'success' : 'default'}
                      />
                    </TableCell>
                    <TableCell sx={{ maxWidth: 300, fontSize: '0.8rem' }}>{role.responsibilities}</TableCell>
                    <TableCell>{role.frequency}</TableCell>
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

export default Governance;
