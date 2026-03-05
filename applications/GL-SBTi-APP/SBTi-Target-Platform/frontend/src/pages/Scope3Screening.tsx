/**
 * Scope3Screening - Scope 3 trigger assessment, category waterfall, hotspot heatmap, coverage calculator.
 *
 * Determines whether Scope 3 targets are required (>=40% of total emissions)
 * and identifies material categories for target coverage.
 */

import React, { useEffect, useMemo } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Alert, Chip, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  LinearProgress,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { PlayArrow, CheckCircle, Warning } from '@mui/icons-material';
import TriggerAssessment from '../components/scope3/TriggerAssessment';
import CategoryWaterfall from '../components/scope3/CategoryWaterfall';
import HotspotHeatmap from '../components/scope3/HotspotHeatmap';
import CoverageCalculator from '../components/scope3/CoverageCalculator';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchScope3Screening, runScope3Trigger, fetchScope3Coverage, fetchScope3Hotspots,
  selectScope3Screening, selectScope3Coverage, selectScope3Hotspots, selectScope3Loading,
} from '../store/slices/scope3Slice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_SCREENING = {
  id: 'sc3_1',
  organization_id: 'org_default',
  total_scope_1_2: 60000,
  total_scope_3: 168000,
  scope_3_pct: 73.7,
  trigger_met: true,
  categories: [
    { category_number: 1, category_name: 'Purchased Goods & Services', emissions: 68000, pct_of_scope3: 40.5, included: true, significance: 'high' },
    { category_number: 4, category_name: 'Upstream Transportation', emissions: 25000, pct_of_scope3: 14.9, included: true, significance: 'high' },
    { category_number: 11, category_name: 'Use of Sold Products', emissions: 22000, pct_of_scope3: 13.1, included: true, significance: 'high' },
    { category_number: 6, category_name: 'Business Travel', emissions: 15000, pct_of_scope3: 8.9, included: true, significance: 'medium' },
    { category_number: 7, category_name: 'Employee Commuting', emissions: 12000, pct_of_scope3: 7.1, included: true, significance: 'medium' },
    { category_number: 5, category_name: 'Waste Generated', emissions: 8000, pct_of_scope3: 4.8, included: false, significance: 'low' },
    { category_number: 2, category_name: 'Capital Goods', emissions: 7000, pct_of_scope3: 4.2, included: false, significance: 'low' },
    { category_number: 3, category_name: 'Fuel & Energy Activities', emissions: 5000, pct_of_scope3: 3.0, included: false, significance: 'low' },
    { category_number: 9, category_name: 'Downstream Transportation', emissions: 3500, pct_of_scope3: 2.1, included: false, significance: 'negligible' },
    { category_number: 12, category_name: 'End-of-Life Treatment', emissions: 2500, pct_of_scope3: 1.5, included: false, significance: 'negligible' },
  ],
  two_thirds_coverage_met: true,
  current_coverage_pct: 84.5,
  screening_date: '2025-02-20',
};

const SIGNIFICANCE_COLORS: Record<string, string> = {
  high: '#C62828', medium: '#EF6C00', low: '#1B5E20', negligible: '#9E9E9E',
};

const Scope3Screening: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const screening = useAppSelector(selectScope3Screening);
  const loading = useAppSelector(selectScope3Loading);

  useEffect(() => {
    dispatch(fetchScope3Screening(orgId));
    dispatch(fetchScope3Coverage(orgId));
    dispatch(fetchScope3Hotspots(orgId));
  }, [dispatch, orgId]);

  const data = screening || DEMO_SCREENING;

  const waterfallData = data.categories
    .sort((a, b) => b.emissions - a.emissions)
    .map((c) => ({
      name: `Cat ${c.category_number}`,
      fullName: c.category_name,
      emissions: c.emissions,
      pct: c.pct_of_scope3,
      included: c.included,
      significance: c.significance,
    }));

  const scopePie = [
    { name: 'Scope 1+2', value: data.total_scope_1_2, color: '#1B5E20' },
    { name: 'Scope 3', value: data.total_scope_3, color: '#EF6C00' },
  ];

  if (loading && !screening) return <LoadingSpinner message="Loading Scope 3 screening..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Scope 3 Screening</Typography>
          <Typography variant="body2" color="text.secondary">
            Trigger assessment and category coverage analysis for Scope 3 target setting
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<PlayArrow />} onClick={() => dispatch(runScope3Trigger(orgId))}>
          Run Screening
        </Button>
      </Box>

      {/* Trigger Assessment */}
      <Alert severity={data.trigger_met ? 'warning' : 'success'} sx={{ mb: 3 }}>
        {data.trigger_met
          ? `Scope 3 target required: Scope 3 emissions represent ${data.scope_3_pct.toFixed(1)}% of total emissions (threshold: 40%)`
          : `Scope 3 target not required: Scope 3 emissions represent ${data.scope_3_pct.toFixed(1)}% of total emissions (below 40% threshold)`
        }
      </Alert>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Scope 3 Share</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: data.scope_3_pct >= 40 ? '#C62828' : '#2E7D32' }}>
                {data.scope_3_pct.toFixed(0)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">of total emissions</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Total Scope 3</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {(data.total_scope_3 / 1000).toFixed(0)}K
              </Typography>
              <Typography variant="caption" color="text.secondary">tCO2e</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Coverage</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: data.two_thirds_coverage_met ? '#2E7D32' : '#C62828' }}>
                {data.current_coverage_pct.toFixed(0)}%
              </Typography>
              <Chip
                label={data.two_thirds_coverage_met ? 'Meets 67%' : 'Below 67%'}
                size="small"
                color={data.two_thirds_coverage_met ? 'success' : 'error'}
                sx={{ mt: 0.5 }}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Categories Included</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700 }}>
                {data.categories.filter((c) => c.included).length}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                of {data.categories.length} screened
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Waterfall / Bar */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Scope 3 Category Breakdown</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={waterfallData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" fontSize={10} />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip
                    formatter={(value: number, name: string, entry: any) => [
                      `${value.toLocaleString()} tCO2e (${entry.payload.pct.toFixed(1)}%)`,
                      entry.payload.fullName,
                    ]}
                  />
                  <Bar dataKey="emissions" name="Emissions">
                    {waterfallData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.included ? '#1B5E20' : '#BDBDBD'} opacity={entry.included ? 1 : 0.5} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', gap: 2, mt: 1, justifyContent: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 12, height: 12, backgroundColor: '#1B5E20', borderRadius: 1 }} />
                  <Typography variant="caption">Included in target</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 12, height: 12, backgroundColor: '#BDBDBD', borderRadius: 1 }} />
                  <Typography variant="caption">Excluded</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Scope Split Pie */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emissions Scope Split</Typography>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={scopePie} cx="50%" cy="50%" innerRadius={50} outerRadius={75} dataKey="value" label={({ name, value }) => `${name}: ${(value / 1000).toFixed(0)}K`}>
                    {scopePie.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}
                  </Pie>
                  <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Category Detail Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Category Screening Detail</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Cat #</TableCell>
                  <TableCell>Category Name</TableCell>
                  <TableCell align="right">Emissions (tCO2e)</TableCell>
                  <TableCell align="center">% of Scope 3</TableCell>
                  <TableCell align="center">Significance</TableCell>
                  <TableCell align="center">Included</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.categories.sort((a, b) => b.emissions - a.emissions).map((cat) => (
                  <TableRow key={cat.category_number} hover sx={{ opacity: cat.included ? 1 : 0.6 }}>
                    <TableCell sx={{ fontWeight: 600 }}>{cat.category_number}</TableCell>
                    <TableCell>{cat.category_name}</TableCell>
                    <TableCell align="right">{cat.emissions.toLocaleString()}</TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={cat.pct_of_scope3}
                          sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption" sx={{ minWidth: 35 }}>{cat.pct_of_scope3.toFixed(1)}%</Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={cat.significance}
                        size="small"
                        sx={{ fontSize: '0.65rem', textTransform: 'capitalize', backgroundColor: SIGNIFICANCE_COLORS[cat.significance] + '22', color: SIGNIFICANCE_COLORS[cat.significance] }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      {cat.included ? <CheckCircle color="success" fontSize="small" /> : <Warning color="disabled" fontSize="small" />}
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

export default Scope3Screening;
