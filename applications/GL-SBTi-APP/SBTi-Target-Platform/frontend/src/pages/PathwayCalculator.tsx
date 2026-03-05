/**
 * PathwayCalculator - ACA & SDA pathway calculation with comparison charts.
 *
 * Calculates emission reduction pathways, compares alignment scenarios,
 * and displays milestone markers along the decarbonization path.
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Grid, Box, Typography, Card, CardContent, Alert, Chip, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';
import PathwayChart from '../components/pathways/PathwayChart';
import ACACalculator from '../components/pathways/ACACalculator';
import SDASelector from '../components/pathways/SDASelector';
import PathwayComparison from '../components/pathways/PathwayComparison';
import MilestoneMarkers from '../components/pathways/MilestoneMarkers';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchPathway, fetchComparisons, selectCurrentPathway, selectPathwayComparisons, selectPathwayLoading } from '../store/slices/pathwaySlice';
import { selectTargets } from '../store/slices/targetSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_PATHWAY = {
  id: 'pw_1',
  target_id: 'tgt_1',
  method: 'aca' as const,
  alignment: '1.5C' as const,
  base_year: 2019,
  target_year: 2030,
  base_emissions: 90000,
  annual_reduction_rate: 4.2,
  milestones: [
    { year: 2019, emissions: 90000, reduction_pct: 0, on_track: true },
    { year: 2022, emissions: 78660, reduction_pct: 12.6, on_track: true },
    { year: 2025, emissions: 68723, reduction_pct: 23.6, on_track: true },
    { year: 2027, emissions: 62957, reduction_pct: 30.0, on_track: true },
    { year: 2030, emissions: 52200, reduction_pct: 42.0, on_track: true },
  ],
  parameters: { method: 'aca' as const, convergence_year: 2050, budget_approach: 'linear', sector: null, region: 'global' },
  created_at: '2024-06-15',
};

const DEMO_COMPARISONS = [
  { scenario: '1.5C', pathway_data: [{ year: 2019, emissions: 90000 }, { year: 2025, emissions: 67320 }, { year: 2030, emissions: 52200 }], annual_rate: 4.2, alignment: '1.5C' as const },
  { scenario: 'WB2C', pathway_data: [{ year: 2019, emissions: 90000 }, { year: 2025, emissions: 76500 }, { year: 2030, emissions: 67500 }], annual_rate: 2.5, alignment: 'WB2C' as const },
  { scenario: '2C', pathway_data: [{ year: 2019, emissions: 90000 }, { year: 2025, emissions: 79200 }, { year: 2030, emissions: 72000 }], annual_rate: 1.8, alignment: '2C' as const },
];

const PathwayCalculator: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const targets = useAppSelector(selectTargets);
  const currentPathway = useAppSelector(selectCurrentPathway);
  const comparisons = useAppSelector(selectPathwayComparisons);
  const loading = useAppSelector(selectPathwayLoading);
  const [selectedTargetId, setSelectedTargetId] = useState('tgt_1');
  const [method, setMethod] = useState<'aca' | 'sda'>('aca');

  useEffect(() => {
    if (selectedTargetId) {
      dispatch(fetchPathway(selectedTargetId));
      dispatch(fetchComparisons(selectedTargetId));
    }
  }, [dispatch, selectedTargetId]);

  const pathway = currentPathway || DEMO_PATHWAY;
  const compData = comparisons.length > 0 ? comparisons : DEMO_COMPARISONS;

  const fullPathwayData = useMemo(() => {
    const points: { year: number; actual: number; pathway: number; budget: number }[] = [];
    for (let y = pathway.base_year; y <= pathway.target_year; y++) {
      const t = y - pathway.base_year;
      const pathwayEmissions = pathway.base_emissions * Math.pow(1 - pathway.annual_reduction_rate / 100, t);
      const actual = y <= 2025 ? pathway.base_emissions * Math.pow(0.96, t) : 0;
      points.push({
        year: y,
        actual: y <= 2025 ? Math.round(actual) : 0,
        pathway: Math.round(pathwayEmissions),
        budget: Math.round(pathway.base_emissions * (1 - (t / (pathway.target_year - pathway.base_year)) * (pathway.annual_reduction_rate * (pathway.target_year - pathway.base_year) / 100))),
      });
    }
    return points;
  }, [pathway]);

  if (loading && !currentPathway) return <LoadingSpinner message="Loading pathway data..." />;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Pathway Calculator</Typography>
        <Typography variant="body2" color="text.secondary">
          Calculate and compare emission reduction pathways using ACA or SDA methods
        </Typography>
      </Box>

      {/* Method Selection */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Target</InputLabel>
            <Select value={selectedTargetId} label="Target" onChange={(e: SelectChangeEvent) => setSelectedTargetId(e.target.value)}>
              <MenuItem value="tgt_1">Near-term Scope 1+2 Absolute</MenuItem>
              <MenuItem value="tgt_2">Near-term Scope 3 Absolute</MenuItem>
              <MenuItem value="tgt_3">Long-term Net-Zero S1+2</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label="ACA (Absolute Contraction)"
              color={method === 'aca' ? 'primary' : 'default'}
              onClick={() => setMethod('aca')}
              sx={{ cursor: 'pointer' }}
            />
            <Chip
              label="SDA (Sectoral Decarbonization)"
              color={method === 'sda' ? 'primary' : 'default'}
              onClick={() => setMethod('sda')}
              sx={{ cursor: 'pointer' }}
            />
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Alert severity="info" sx={{ py: 0.5 }}>
            Min rate: {pathway.alignment === '1.5C' ? '4.2%' : pathway.alignment === 'WB2C' ? '2.5%' : '1.8%'}/yr for {pathway.alignment}
          </Alert>
        </Grid>
      </Grid>

      {/* Pathway Chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Emission Reduction Pathway</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={fullPathwayData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip formatter={(value: number) => [value > 0 ? value.toLocaleString() : '--', '']} />
                  <Legend />
                  <Area type="monotone" dataKey="pathway" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.15} name="Pathway Envelope" />
                  <Line type="monotone" dataKey="pathway" stroke="#1B5E20" strokeWidth={2} strokeDasharray="5 5" name="Target Pathway" dot={false} />
                  <Line type="monotone" dataKey="actual" stroke="#0D47A1" strokeWidth={2.5} name="Actual Emissions" connectNulls={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Pathway Parameters</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Method</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.method.toUpperCase()}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Alignment</Typography>
                  <Chip label={pathway.alignment} size="small" color={pathway.alignment === '1.5C' ? 'success' : 'warning'} />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Annual Reduction</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.annual_reduction_rate}%</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Base Year</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.base_year}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Target Year</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.target_year}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Base Emissions</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.base_emissions.toLocaleString()} tCO2e</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Target Emissions</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{pathway.milestones[pathway.milestones.length - 1].emissions.toLocaleString()} tCO2e</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <MilestoneMarkers milestones={pathway.milestones as any} />
        </Grid>
      </Grid>

      {/* Comparison */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          {method === 'aca' ? (
            <ACACalculator onCalculate={(params) => console.log('ACA calc:', params)} calculating={false} />
          ) : (
            <SDASelector onSelect={(params) => console.log('SDA select:', params)} />
          )}
        </Grid>
        <Grid item xs={12} md={6}>
          <PathwayComparison comparisons={compData as any} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default PathwayCalculator;
