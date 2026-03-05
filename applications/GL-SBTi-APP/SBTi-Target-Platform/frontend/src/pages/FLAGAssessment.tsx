/**
 * FLAGAssessment - Forest, Land and Agriculture target assessment.
 *
 * Determines FLAG target requirements, commodity-level emissions,
 * deforestation tracking, and FLAG/non-FLAG emissions split.
 */

import React, { useEffect } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Alert, Chip, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { PlayArrow, Forest, Agriculture } from '@mui/icons-material';
import FLAGTrigger from '../components/flag/FLAGTrigger';
import CommoditySelector from '../components/flag/CommoditySelector';
import DeforestationTracker from '../components/flag/DeforestationTracker';
import EmissionsSplit from '../components/flag/EmissionsSplit';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchFLAGAssessment, fetchEmissionsSplit, selectFLAGAssessment, selectEmissionsSplit, selectFLAGLoading } from '../store/slices/flagSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_ASSESSMENT = {
  id: 'flag_1',
  organization_id: 'org_default',
  flag_target_required: false,
  total_land_use_emissions: 1200,
  total_emissions: 228000,
  flag_pct: 0.53,
  threshold_pct: 20,
  commodities: [
    { commodity: 'Palm Oil', emissions: 450, deforestation_free: true, certification: 'RSPO' },
    { commodity: 'Soy', emissions: 320, deforestation_free: false, certification: 'None' },
    { commodity: 'Timber/Pulp', emissions: 180, deforestation_free: true, certification: 'FSC' },
    { commodity: 'Beef', emissions: 150, deforestation_free: false, certification: 'None' },
    { commodity: 'Cocoa', emissions: 100, deforestation_free: true, certification: 'Rainforest Alliance' },
  ],
  assessment_date: '2025-02-15',
  sector: 'technology',
};

const DEMO_SPLIT = {
  flag_emissions: 1200,
  non_flag_emissions: 226800,
  flag_pct: 0.53,
  by_commodity: [
    { commodity: 'Palm Oil', emissions: 450 },
    { commodity: 'Soy', emissions: 320 },
    { commodity: 'Timber/Pulp', emissions: 180 },
    { commodity: 'Beef', emissions: 150 },
    { commodity: 'Cocoa', emissions: 100 },
  ],
};

const COMMODITY_COLORS = ['#1B5E20', '#4CAF50', '#81C784', '#A5D6A7', '#C8E6C9'];

const FLAGAssessment: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const assessment = useAppSelector(selectFLAGAssessment);
  const split = useAppSelector(selectEmissionsSplit);
  const loading = useAppSelector(selectFLAGLoading);

  useEffect(() => {
    dispatch(fetchFLAGAssessment(orgId));
    dispatch(fetchEmissionsSplit(orgId));
  }, [dispatch, orgId]);

  const data = assessment || DEMO_ASSESSMENT;
  const splitData = split || DEMO_SPLIT;

  if (loading && !assessment) return <LoadingSpinner message="Loading FLAG assessment..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">FLAG Assessment</Typography>
          <Typography variant="body2" color="text.secondary">
            Forest, Land and Agriculture target requirements and commodity analysis
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<PlayArrow />} onClick={() => dispatch(fetchFLAGAssessment(orgId))}>
          Run Assessment
        </Button>
      </Box>

      {/* Trigger Result */}
      <Alert
        severity={data.flag_target_required ? 'warning' : 'info'}
        sx={{ mb: 3 }}
        icon={<Forest />}
      >
        {data.flag_target_required
          ? `FLAG target required: Land-use emissions represent ${data.flag_pct.toFixed(1)}% of total emissions (threshold: ${data.threshold_pct}%)`
          : `FLAG target not required: Land-use emissions represent ${data.flag_pct.toFixed(1)}% of total emissions (below ${data.threshold_pct}% threshold). Sector: ${data.sector}`
        }
      </Alert>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">FLAG Emissions</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {data.total_land_use_emissions.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="text.secondary">tCO2e</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">FLAG Share</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: data.flag_pct >= 20 ? '#C62828' : '#2E7D32' }}>
                {data.flag_pct.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">of total emissions</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Commodities Tracked</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {data.commodities.length}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {data.commodities.filter((c) => c.deforestation_free).length} deforestation-free
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Target Required</Typography>
              <Chip
                label={data.flag_target_required ? 'Yes' : 'No'}
                color={data.flag_target_required ? 'error' : 'success'}
                sx={{ mt: 1, fontSize: '1.1rem', fontWeight: 700 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Commodity Breakdown */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emissions by Commodity</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={splitData.by_commodity} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={(v) => `${v} t`} />
                  <YAxis type="category" dataKey="commodity" width={100} fontSize={11} />
                  <Tooltip formatter={(value: number) => [`${value.toLocaleString()} tCO2e`, 'Emissions']} />
                  <Bar dataKey="emissions" name="Emissions">
                    {splitData.by_commodity.map((_: any, idx: number) => (
                      <Cell key={idx} fill={COMMODITY_COLORS[idx % COMMODITY_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* FLAG / Non-FLAG Split */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>FLAG vs Non-FLAG Emissions</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'FLAG', value: splitData.flag_emissions },
                      { name: 'Non-FLAG', value: splitData.non_flag_emissions },
                    ]}
                    cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value"
                    label={({ name, value }) => `${name}: ${(value / 1000).toFixed(1)}K`}
                  >
                    <Cell fill="#4CAF50" />
                    <Cell fill="#0D47A1" />
                  </Pie>
                  <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Commodity Detail Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Commodity Detail</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Commodity</TableCell>
                  <TableCell align="right">Emissions (tCO2e)</TableCell>
                  <TableCell align="center">% of FLAG</TableCell>
                  <TableCell align="center">Deforestation-Free</TableCell>
                  <TableCell>Certification</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.commodities.sort((a, b) => b.emissions - a.emissions).map((c) => (
                  <TableRow key={c.commodity} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{c.commodity}</TableCell>
                    <TableCell align="right">{c.emissions.toLocaleString()}</TableCell>
                    <TableCell align="center">
                      {((c.emissions / data.total_land_use_emissions) * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={c.deforestation_free ? 'Yes' : 'No'}
                        size="small"
                        color={c.deforestation_free ? 'success' : 'error'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip label={c.certification} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
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

export default FLAGAssessment;
