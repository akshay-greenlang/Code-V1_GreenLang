/**
 * RiskManagement - TCFD Pillar 3: Risk register, 5x5 heat matrix, response tracker.
 */

import React, { useState, useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, Button, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent, LinearProgress } from '@mui/material';
import { Shield, Warning, CheckCircle, TrendingDown } from '@mui/icons-material';
import DataTable, { Column } from '../components/common/DataTable';
import RiskBadge from '../components/common/RiskBadge';
import StatusBadge from '../components/common/StatusBadge';
import StatCard from '../components/common/StatCard';

interface RiskRecord {
  id: string;
  name: string;
  category: string;
  riskLevel: string;
  likelihood: string;
  impact: string;
  inherentScore: number;
  residualScore: number;
  response: string;
  status: string;
  owner: string;
  ermIntegrated: boolean;
  nextReview: string;
  actions: { description: string; status: string; effectiveness: number }[];
}

const REGISTER: RiskRecord[] = [
  { id: '1', name: 'Carbon pricing regulation', category: 'transition', riskLevel: 'high', likelihood: 'very_likely', impact: 'major', inherentScore: 80, residualScore: 55, response: 'mitigate', status: 'mitigating', owner: 'Michael Park', ermIntegrated: true, nextReview: '2025-06-15', actions: [{ description: 'Scope 1 reduction program', status: 'in_progress', effectiveness: 60 }, { description: 'Carbon offset procurement', status: 'completed', effectiveness: 80 }] },
  { id: '2', name: 'Extreme weather events', category: 'physical', riskLevel: 'critical', likelihood: 'likely', impact: 'catastrophic', inherentScore: 95, residualScore: 70, response: 'transfer', status: 'mitigating', owner: 'Sarah Chen', ermIntegrated: true, nextReview: '2025-04-01', actions: [{ description: 'Insurance coverage upgrade', status: 'in_progress', effectiveness: 45 }, { description: 'Facility hardening', status: 'not_started', effectiveness: 0 }] },
  { id: '3', name: 'Technology disruption', category: 'transition', riskLevel: 'medium', likelihood: 'possible', impact: 'moderate', inherentScore: 50, residualScore: 35, response: 'accept', status: 'monitoring', owner: 'James Mitchell', ermIntegrated: true, nextReview: '2025-09-01', actions: [{ description: 'Technology scouting program', status: 'in_progress', effectiveness: 55 }] },
  { id: '4', name: 'Water stress', category: 'physical', riskLevel: 'high', likelihood: 'likely', impact: 'major', inherentScore: 75, residualScore: 50, response: 'mitigate', status: 'mitigating', owner: 'Michael Park', ermIntegrated: false, nextReview: '2025-07-01', actions: [{ description: 'Water recycling investment', status: 'approved', effectiveness: 0 }] },
  { id: '5', name: 'Greenwashing litigation', category: 'transition', riskLevel: 'medium', likelihood: 'possible', impact: 'major', inherentScore: 60, residualScore: 40, response: 'avoid', status: 'mitigating', owner: 'Aisha Rahman', ermIntegrated: true, nextReview: '2025-05-01', actions: [{ description: 'Claims verification audit', status: 'completed', effectiveness: 90 }] },
  { id: '6', name: 'Supply chain disruption', category: 'physical', riskLevel: 'medium', likelihood: 'possible', impact: 'moderate', inherentScore: 45, residualScore: 30, response: 'mitigate', status: 'monitoring', owner: 'Sarah Chen', ermIntegrated: false, nextReview: '2025-08-01', actions: [{ description: 'Supplier diversification', status: 'in_progress', effectiveness: 50 }] },
];

const HEAT_MAP: number[][] = [
  [0, 0, 0, 1, 0],
  [0, 0, 1, 1, 0],
  [0, 1, 1, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0],
];
const LIKELIHOOD_LABELS = ['Rare', 'Unlikely', 'Possible', 'Likely', 'Almost Certain'];
const IMPACT_LABELS = ['Insignificant', 'Minor', 'Moderate', 'Major', 'Catastrophic'];
const HEAT_COLORS = ['#E8F5E9', '#FFF9C4', '#FFE0B2', '#FFCDD2', '#EF9A9A'];

const RiskManagement: React.FC = () => {
  const [statusFilter, setStatusFilter] = useState('all');

  const filtered = statusFilter === 'all' ? REGISTER : REGISTER.filter((r) => r.status === statusFilter);
  const ermIntegrated = REGISTER.filter((r) => r.ermIntegrated).length;

  const columns: Column<RiskRecord>[] = [
    { id: 'name', label: 'Risk', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'riskLevel', label: 'Level', accessor: (r) => <RiskBadge level={r.riskLevel} />, sortAccessor: (r) => r.inherentScore },
    { id: 'inherentScore', label: 'Inherent', accessor: (r) => r.inherentScore, sortAccessor: (r) => r.inherentScore, align: 'center' },
    { id: 'residualScore', label: 'Residual', accessor: (r) => <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><LinearProgress variant="determinate" value={r.residualScore} sx={{ width: 50, height: 6, borderRadius: 3 }} color={r.residualScore > 60 ? 'error' : r.residualScore > 40 ? 'warning' : 'success'} /><Typography variant="caption">{r.residualScore}</Typography></Box>, sortAccessor: (r) => r.residualScore, align: 'center' },
    { id: 'response', label: 'Response', accessor: (r) => <Chip label={r.response} size="small" variant="outlined" sx={{ textTransform: 'capitalize' }} />, sortAccessor: (r) => r.response },
    { id: 'status', label: 'Status', accessor: (r) => <StatusBadge status={r.status} /> },
    { id: 'owner', label: 'Owner', accessor: (r) => r.owner },
    { id: 'erm', label: 'ERM', accessor: (r) => <Chip label={r.ermIntegrated ? 'Yes' : 'No'} size="small" color={r.ermIntegrated ? 'success' : 'default'} /> },
  ];

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Risk Management</Typography>
        <Typography variant="body2" color="text.secondary">
          TCFD Pillar 3 -- Climate risk identification, assessment, and management processes
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Total Risks" value={REGISTER.length} icon={<Warning />} color="error" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Being Mitigated" value={REGISTER.filter((r) => r.status === 'mitigating').length} icon={<Shield />} color="warning" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="ERM Integrated" value={ermIntegrated} icon={<CheckCircle />} subtitle={`of ${REGISTER.length} total`} color="success" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Avg Residual Score" value={Math.round(REGISTER.reduce((s, r) => s + r.residualScore, 0) / REGISTER.length)} icon={<TrendingDown />} subtitle="Target: < 40" color="info" />
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Heat Map */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Risk Heat Map (5x5)</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5, pr: 1 }}>
                  {IMPACT_LABELS.map((l) => (
                    <Typography key={l} variant="caption" sx={{ width: 60, textAlign: 'center', fontSize: '0.6rem' }}>{l}</Typography>
                  ))}
                </Box>
                {HEAT_MAP.slice().reverse().map((row, rowIdx) => (
                  <Box key={rowIdx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" sx={{ width: 80, textAlign: 'right', pr: 1, fontSize: '0.65rem' }}>
                      {LIKELIHOOD_LABELS[4 - rowIdx]}
                    </Typography>
                    {row.map((count, colIdx) => {
                      const severity = (4 - rowIdx) + colIdx;
                      const bgColor = severity >= 7 ? HEAT_COLORS[4] : severity >= 5 ? HEAT_COLORS[3] : severity >= 3 ? HEAT_COLORS[2] : severity >= 1 ? HEAT_COLORS[1] : HEAT_COLORS[0];
                      return (
                        <Box key={colIdx} sx={{
                          width: 60, height: 48, borderRadius: 1, display: 'flex',
                          alignItems: 'center', justifyContent: 'center',
                          backgroundColor: count > 0 ? bgColor : '#F5F5F5',
                          border: '1px solid #E0E0E0',
                        }}>
                          {count > 0 && <Typography variant="body2" sx={{ fontWeight: 700 }}>{count}</Typography>}
                        </Box>
                      );
                    })}
                  </Box>
                ))}
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">Impact Severity --&gt;</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Response Tracker */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Response Action Tracker</Typography>
              {REGISTER.slice(0, 4).map((risk) => (
                <Box key={risk.id} sx={{ mb: 2, p: 1.5, border: '1px solid #E0E0E0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{risk.name}</Typography>
                    <RiskBadge level={risk.riskLevel} size="small" />
                  </Box>
                  {risk.actions.map((action, idx) => (
                    <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 1, mb: 0.5 }}>
                      <StatusBadge status={action.status} size="small" />
                      <Typography variant="body2" sx={{ flex: 1, fontSize: '0.8rem' }}>{action.description}</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <LinearProgress variant="determinate" value={action.effectiveness} sx={{ width: 40, height: 5, borderRadius: 3 }} />
                        <Typography variant="caption">{action.effectiveness}%</Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mb: 2 }}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Status</InputLabel>
          <Select value={statusFilter} label="Status" onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}>
            <MenuItem value="all">All Statuses</MenuItem>
            <MenuItem value="open">Open</MenuItem>
            <MenuItem value="mitigating">Mitigating</MenuItem>
            <MenuItem value="monitoring">Monitoring</MenuItem>
            <MenuItem value="closed">Closed</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <DataTable columns={columns} data={filtered} keyAccessor={(r) => r.id} defaultSortColumn="inherentScore" defaultSortDirection="desc" />
    </Box>
  );
};

export default RiskManagement;
