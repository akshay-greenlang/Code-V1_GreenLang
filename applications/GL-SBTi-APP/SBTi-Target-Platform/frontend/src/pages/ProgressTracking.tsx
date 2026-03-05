/**
 * ProgressTracking - Annual progress monitoring against SBTi targets.
 *
 * Displays emissions vs pathway, RAG indicators, trend arrows, scope breakdown,
 * and forward projection charts.
 */

import React, { useEffect, useMemo } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
} from '@mui/material';
import {
  ComposedChart, Line, Bar, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import EmissionsVsPathway from '../components/progress/EmissionsVsPathway';
import RAGIndicator from '../components/progress/RAGIndicator';
import ScopeBreakdown from '../components/progress/ScopeBreakdown';
import ProjectionChart from '../components/progress/ProjectionChart';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchProgressDashboard, selectProgressSummaries, selectProgressLoading } from '../store/slices/progressSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_PROGRESS = [
  { target_id: '1', target_name: 'Near-term S1+2 Absolute', year: 2024, actual_emissions: 60000, pathway_emissions: 62957, variance_pct: -4.7, rag_status: 'green' as const, cumulative_reduction_pct: 33.3, annual_change_pct: -5.2 },
  { target_id: '2', target_name: 'Near-term S3 Absolute', year: 2024, actual_emissions: 175000, pathway_emissions: 172000, variance_pct: 1.7, rag_status: 'amber' as const, cumulative_reduction_pct: 16.7, annual_change_pct: -3.8 },
  { target_id: '3', target_name: 'Long-term Net-Zero S1+2', year: 2024, actual_emissions: 60000, pathway_emissions: 85500, variance_pct: -29.8, rag_status: 'green' as const, cumulative_reduction_pct: 33.3, annual_change_pct: -5.2 },
];

const DEMO_HISTORY = [
  { year: 2019, scope_1: 55000, scope_2: 35000, scope_3: 210000, total: 300000, pathway: 300000 },
  { year: 2020, scope_1: 52000, scope_2: 32000, scope_3: 200000, total: 284000, pathway: 287400 },
  { year: 2021, scope_1: 49000, scope_2: 29000, scope_3: 190000, total: 268000, pathway: 274800 },
  { year: 2022, scope_1: 46000, scope_2: 26000, scope_3: 183000, total: 255000, pathway: 262200 },
  { year: 2023, scope_1: 43000, scope_2: 23000, scope_3: 175000, total: 241000, pathway: 249600 },
  { year: 2024, scope_1: 40000, scope_2: 20000, scope_3: 168000, total: 228000, pathway: 237000 },
];

const SCOPE_PIE = [
  { name: 'Scope 1', value: 40000, color: '#1B5E20' },
  { name: 'Scope 2', value: 20000, color: '#0D47A1' },
  { name: 'Scope 3', value: 168000, color: '#EF6C00' },
];

const PROJECTION = [
  { year: 2024, actual: 228000, projected: 228000, pathway: 237000 },
  { year: 2025, actual: 0, projected: 216600, pathway: 224400 },
  { year: 2026, actual: 0, projected: 205770, pathway: 211800 },
  { year: 2027, actual: 0, projected: 195482, pathway: 199200 },
  { year: 2028, actual: 0, projected: 185708, pathway: 186600 },
  { year: 2029, actual: 0, projected: 176422, pathway: 174000 },
  { year: 2030, actual: 0, projected: 167601, pathway: 161400 },
];

const ProgressTracking: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const summaries = useAppSelector(selectProgressSummaries);
  const loading = useAppSelector(selectProgressLoading);

  useEffect(() => {
    dispatch(fetchProgressDashboard(orgId));
  }, [dispatch, orgId]);

  const progress = summaries.length > 0 ? summaries : DEMO_PROGRESS;

  const getRAGColor = (rag: string) => rag === 'green' ? '#2E7D32' : rag === 'amber' ? '#EF6C00' : '#C62828';
  const getTrendIcon = (change: number) => {
    if (change < -2) return <TrendingDown color="success" fontSize="small" />;
    if (change > 2) return <TrendingUp color="error" fontSize="small" />;
    return <TrendingFlat color="action" fontSize="small" />;
  };

  if (loading && summaries.length === 0) return <LoadingSpinner message="Loading progress..." />;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Progress Tracking</Typography>
        <Typography variant="body2" color="text.secondary">
          Monitor annual progress against SBTi emission reduction pathways
        </Typography>
      </Box>

      {/* Target Progress Table */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Target Progress Summary (FY2024)</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Target</TableCell>
                  <TableCell align="right">Actual (tCO2e)</TableCell>
                  <TableCell align="right">Pathway (tCO2e)</TableCell>
                  <TableCell align="center">Variance</TableCell>
                  <TableCell align="center">RAG</TableCell>
                  <TableCell align="center">Cumulative Reduction</TableCell>
                  <TableCell align="center">YoY Change</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {progress.map((p: any) => (
                  <TableRow key={p.target_id} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{p.target_name}</TableCell>
                    <TableCell align="right">{p.actual_emissions.toLocaleString()}</TableCell>
                    <TableCell align="right">{p.pathway_emissions.toLocaleString()}</TableCell>
                    <TableCell align="center">
                      <Typography
                        variant="body2"
                        sx={{ fontWeight: 600, color: p.variance_pct <= 0 ? 'success.main' : 'error.main' }}
                      >
                        {p.variance_pct > 0 ? '+' : ''}{p.variance_pct.toFixed(1)}%
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: getRAGColor(p.rag_status), mx: 'auto' }} />
                    </TableCell>
                    <TableCell align="center">
                      <Chip label={`${p.cumulative_reduction_pct.toFixed(1)}%`} size="small" color="primary" variant="outlined" />
                    </TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                        {getTrendIcon(p.annual_change_pct)}
                        <Typography variant="body2">{p.annual_change_pct.toFixed(1)}%</Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Emissions vs Pathway */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emissions vs Pathway Trend</Typography>
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={DEMO_HISTORY}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value: number) => [value.toLocaleString(), '']} />
                  <Legend />
                  <Bar dataKey="scope_1" stackId="a" fill="#1B5E20" name="Scope 1" />
                  <Bar dataKey="scope_2" stackId="a" fill="#0D47A1" name="Scope 2" />
                  <Bar dataKey="scope_3" stackId="a" fill="#EF6C00" name="Scope 3" />
                  <Line type="monotone" dataKey="pathway" stroke="#C62828" strokeWidth={2} strokeDasharray="5 5" name="Pathway" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Scope Breakdown Pie */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Current Scope Breakdown</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={SCOPE_PIE} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2} dataKey="value"
                    label={({ name, value }) => `${name}: ${(value / 1000).toFixed(0)}K`}
                  >
                    {SCOPE_PIE.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ mt: 1 }}>
                {SCOPE_PIE.map((s) => (
                  <Box key={s.name} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: s.color }} />
                      <Typography variant="caption">{s.name}</Typography>
                    </Box>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {((s.value / 228000) * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Projection */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Forward Projection to Target Year</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={PROJECTION}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
              <Tooltip formatter={(value: number) => [value > 0 ? value.toLocaleString() : '--', '']} />
              <Legend />
              <Area type="monotone" dataKey="projected" stroke="#0D47A1" fill="#0D47A1" fillOpacity={0.15} name="Projected" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="pathway" stroke="#C62828" strokeWidth={2} strokeDasharray="5 5" name="Pathway Target" dot={false} />
              <Line type="monotone" dataKey="actual" stroke="#1B5E20" strokeWidth={2.5} name="Actual" connectNulls={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProgressTracking;
