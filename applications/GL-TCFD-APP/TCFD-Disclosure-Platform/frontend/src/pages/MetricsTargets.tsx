/**
 * MetricsTargets - TCFD Pillar 4: Scope 1/2/3 emissions, intensity trends, target progress, peer benchmarking.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line, PieChart, Pie, Cell,
} from 'recharts';
import { Assessment, TrendingDown, Flag, CompareArrows } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';

/* ── Demo Data ────────────────────────────────────────────────── */

const EMISSIONS_SUMMARY = [
  { scope: 'Scope 1', current: 12500, previous: 13800, target: 10000, color: '#1B5E20' },
  { scope: 'Scope 2 (Market)', current: 8200, previous: 9500, target: 6000, color: '#0D47A1' },
  { scope: 'Scope 2 (Location)', current: 9100, previous: 10200, target: 7000, color: '#1565C0' },
  { scope: 'Scope 3', current: 85000, previous: 88000, target: 68000, color: '#4527A0' },
];

const SCOPE3_BREAKDOWN = [
  { category: 'Cat 1: Purchased Goods', value: 32000, pct: 37.6 },
  { category: 'Cat 4: Upstream Transport', value: 12500, pct: 14.7 },
  { category: 'Cat 11: Use of Sold Products', value: 15200, pct: 17.9 },
  { category: 'Cat 6: Business Travel', value: 4800, pct: 5.6 },
  { category: 'Cat 7: Employee Commuting', value: 3200, pct: 3.8 },
  { category: 'Cat 15: Investments', value: 8500, pct: 10.0 },
  { category: 'Other Categories', value: 8800, pct: 10.4 },
];

const SCOPE3_COLORS = ['#7B1FA2', '#9C27B0', '#AB47BC', '#CE93D8', '#E1BEE7', '#F3E5F5', '#EDE7F6'];

const INTENSITY_TREND = [
  { year: 2019, revenue: 185, fte: 18.5, sqm: 42 },
  { year: 2020, revenue: 170, fte: 17.2, sqm: 40 },
  { year: 2021, revenue: 155, fte: 15.8, sqm: 37 },
  { year: 2022, revenue: 140, fte: 14.5, sqm: 34 },
  { year: 2023, revenue: 125, fte: 13.0, sqm: 30 },
  { year: 2024, revenue: 112, fte: 11.8, sqm: 27 },
];

const TARGETS = [
  { id: '1', name: 'Net Zero by 2050', scope: 'All Scopes', baseYear: 2019, targetYear: 2050, baselineEmissions: 115000, targetEmissions: 0, currentEmissions: 105700, sbtiValidated: true, status: 'on_track' },
  { id: '2', name: '42% Scope 1+2 Reduction', scope: 'Scope 1+2', baseYear: 2019, targetYear: 2030, baselineEmissions: 23300, targetEmissions: 13514, currentEmissions: 20700, sbtiValidated: true, status: 'off_track' },
  { id: '3', name: '25% Scope 3 Reduction', scope: 'Scope 3', baseYear: 2019, targetYear: 2030, baselineEmissions: 91700, targetEmissions: 68775, currentEmissions: 85000, sbtiValidated: true, status: 'on_track' },
  { id: '4', name: '100% Renewable Electricity', scope: 'Scope 2', baseYear: 2020, targetYear: 2030, baselineEmissions: 10200, targetEmissions: 0, currentEmissions: 8200, sbtiValidated: false, status: 'on_track' },
  { id: '5', name: 'Fleet Electrification 80%', scope: 'Scope 1', baseYear: 2021, targetYear: 2028, baselineEmissions: 4500, targetEmissions: 900, currentEmissions: 3200, sbtiValidated: false, status: 'behind' },
];

const PEER_BENCHMARKS = [
  { company: 'Our Company', scope1: 12500, scope2: 8200, scope3: 85000, intensity: 112, sbti: true },
  { company: 'Peer A', scope1: 15800, scope2: 6100, scope3: 92000, intensity: 135, sbti: true },
  { company: 'Peer B', scope1: 9200, scope2: 4500, scope3: 78000, intensity: 98, sbti: true },
  { company: 'Peer C', scope1: 18500, scope2: 11200, scope3: 110000, intensity: 165, sbti: false },
  { company: 'Industry Avg', scope1: 14000, scope2: 7500, scope3: 91000, intensity: 128, sbti: false },
];

/* ── Helpers ───────────────────────────────────────────────────── */

const getTargetProgress = (baseline: number, target: number, current: number): number => {
  if (baseline === target) return 100;
  const totalReduction = baseline - target;
  const achievedReduction = baseline - current;
  return Math.min(100, Math.max(0, (achievedReduction / totalReduction) * 100));
};

const getStatusColor = (status: string): 'success' | 'warning' | 'error' => {
  if (status === 'on_track') return 'success';
  if (status === 'behind') return 'warning';
  return 'error';
};

/* ── Component ─────────────────────────────────────────────────── */

const MetricsTargets: React.FC = () => {
  const [intensityMetric, setIntensityMetric] = useState<'revenue' | 'fte' | 'sqm'>('revenue');

  const totalEmissions = EMISSIONS_SUMMARY[0].current + EMISSIONS_SUMMARY[1].current + EMISSIONS_SUMMARY[3].current;
  const prevTotal = EMISSIONS_SUMMARY[0].previous + EMISSIONS_SUMMARY[1].previous + EMISSIONS_SUMMARY[3].previous;
  const yoyChange = ((totalEmissions - prevTotal) / prevTotal) * 100;

  const intensityLabels: Record<string, string> = {
    revenue: 'tCO2e / $M Revenue',
    fte: 'tCO2e / FTE',
    sqm: 'kgCO2e / sqm',
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Metrics & Targets</Typography>
        <Typography variant="body2" color="text.secondary">
          TCFD Pillar 4 -- GHG emissions, intensity metrics, targets, and peer benchmarking
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Emissions"
            value={totalEmissions}
            icon={<Assessment />}
            subtitle="tCO2e (S1+S2+S3)"
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Year-over-Year"
            value={parseFloat(yoyChange.toFixed(1))}
            format="percent"
            icon={<TrendingDown />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Targets Set"
            value={TARGETS.length}
            icon={<Flag />}
            subtitle={`${TARGETS.filter((t) => t.sbtiValidated).length} SBTi validated`}
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="On Track"
            value={TARGETS.filter((t) => t.status === 'on_track').length}
            icon={<CompareArrows />}
            subtitle={`of ${TARGETS.length} targets`}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Emissions Summary + Scope 3 Breakdown */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>GHG Emissions Summary (tCO2e)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={EMISSIONS_SUMMARY}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="scope" fontSize={11} />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip formatter={(v: number) => [`${v.toLocaleString()} tCO2e`, '']} />
                  <Legend />
                  <Bar dataKey="previous" name="Previous Year" fill="#BDBDBD" />
                  <Bar dataKey="current" name="Current Year">
                    {EMISSIONS_SUMMARY.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Bar>
                  <Bar dataKey="target" name="Target" fill="#E0E0E0" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Scope 3 Breakdown by Category</Typography>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie
                    data={SCOPE3_BREAKDOWN}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={90}
                    dataKey="value"
                    nameKey="category"
                    label={({ pct }: { pct: number }) => `${pct}%`}
                  >
                    {SCOPE3_BREAKDOWN.map((_, idx) => (
                      <Cell key={idx} fill={SCOPE3_COLORS[idx]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v: number) => [`${v.toLocaleString()} tCO2e`, '']} />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                {SCOPE3_BREAKDOWN.map((item, idx) => (
                  <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: SCOPE3_COLORS[idx] }} />
                    <Typography variant="caption" sx={{ fontSize: '0.6rem' }}>{item.category.split(':')[0]}</Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Intensity Trend */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Emissions Intensity Trend</Typography>
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Intensity Metric</InputLabel>
              <Select
                value={intensityMetric}
                label="Intensity Metric"
                onChange={(e: SelectChangeEvent) => setIntensityMetric(e.target.value as 'revenue' | 'fte' | 'sqm')}
              >
                <MenuItem value="revenue">Per $M Revenue</MenuItem>
                <MenuItem value="fte">Per FTE</MenuItem>
                <MenuItem value="sqm">Per sqm</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={INTENSITY_TREND}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis
                label={{ value: intensityLabels[intensityMetric], angle: -90, position: 'insideLeft', offset: 10 }}
              />
              <Tooltip formatter={(v: number) => [`${v} ${intensityLabels[intensityMetric]}`, 'Intensity']} />
              <Line
                type="monotone"
                dataKey={intensityMetric}
                stroke="#1B5E20"
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Targets Progress */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Target Progress Tracker</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Target</TableCell>
                  <TableCell>Scope</TableCell>
                  <TableCell align="center">Base Year</TableCell>
                  <TableCell align="center">Target Year</TableCell>
                  <TableCell align="center">Progress</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell align="center">SBTi</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {TARGETS.map((target) => {
                  const progress = getTargetProgress(target.baselineEmissions, target.targetEmissions, target.currentEmissions);
                  return (
                    <TableRow key={target.id} hover>
                      <TableCell sx={{ fontWeight: 500 }}>{target.name}</TableCell>
                      <TableCell>
                        <Chip label={target.scope} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell align="center">{target.baseYear}</TableCell>
                      <TableCell align="center">{target.targetYear}</TableCell>
                      <TableCell align="center" sx={{ minWidth: 160 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{ flex: 1, height: 8, borderRadius: 4 }}
                            color={progress >= 60 ? 'success' : progress >= 30 ? 'warning' : 'error'}
                          />
                          <Typography variant="caption" sx={{ fontWeight: 600, minWidth: 35 }}>
                            {progress.toFixed(0)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={target.status.replace(/_/g, ' ')}
                          size="small"
                          color={getStatusColor(target.status)}
                          sx={{ textTransform: 'capitalize' }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={target.sbtiValidated ? 'Validated' : 'Pending'}
                          size="small"
                          color={target.sbtiValidated ? 'success' : 'default'}
                          variant={target.sbtiValidated ? 'filled' : 'outlined'}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Peer Benchmarking */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Peer Benchmarking Comparison</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={7}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={PEER_BENCHMARKS}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="company" fontSize={11} />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip formatter={(v: number) => [`${v.toLocaleString()} tCO2e`, '']} />
                  <Legend />
                  <Bar dataKey="scope1" name="Scope 1" fill="#1B5E20" />
                  <Bar dataKey="scope2" name="Scope 2" fill="#0D47A1" />
                  <Bar dataKey="scope3" name="Scope 3" fill="#4527A0" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
            <Grid item xs={12} md={5}>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Company</TableCell>
                      <TableCell align="right">Intensity</TableCell>
                      <TableCell align="center">SBTi</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {PEER_BENCHMARKS.map((peer) => (
                      <TableRow
                        key={peer.company}
                        hover
                        sx={{ backgroundColor: peer.company === 'Our Company' ? '#E8F5E9' : undefined }}
                      >
                        <TableCell sx={{ fontWeight: peer.company === 'Our Company' ? 700 : 400 }}>
                          {peer.company}
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: 500,
                              color: peer.intensity <= 112 ? 'success.main' : 'text.primary',
                            }}
                          >
                            {peer.intensity} tCO2e/$M
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={peer.sbti ? 'Yes' : 'No'}
                            size="small"
                            color={peer.sbti ? 'success' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MetricsTargets;
