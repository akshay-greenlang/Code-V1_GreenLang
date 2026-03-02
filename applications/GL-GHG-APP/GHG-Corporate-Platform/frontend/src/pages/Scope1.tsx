/**
 * Scope 1 Page - Direct emissions overview
 *
 * Composes a Scope 1 total stat card, SourceBreakdown chart,
 * FacilityTable, GasBreakdown chart, and an "Add Data" button.
 */

import React, { useEffect } from 'react';
import { Grid, Box, Typography, Button, Alert, Card, CardContent } from '@mui/material';
import { Add, Factory } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import { fetchScope1Summary } from '../store/slices/scope1Slice';
import StatCard from '../components/common/StatCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import SourceBreakdown from '../components/scope1/SourceBreakdown';
import FacilityTable from '../components/scope1/FacilityTable';
import GasBreakdown from '../components/scope1/GasBreakdown';
import { formatEmissions } from '../utils/formatters';
import type { FacilityEmissions } from '../types';

const DEMO_INVENTORY_ID = 'demo-inventory';

const Scope1Page: React.FC = () => {
  const dispatch = useAppDispatch();
  const { summary, categories, loading, error } = useAppSelector((state) => state.scope1);

  useEffect(() => {
    dispatch(fetchScope1Summary(DEMO_INVENTORY_ID));
  }, [dispatch]);

  if (loading && !summary) return <LoadingSpinner message="Loading Scope 1 data..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const facilities: FacilityEmissions[] = (summary?.by_entity ?? []).map((e) => ({
    facility_id: e.entity_id,
    facility_name: e.entity_name,
    country: '',
    source_categories: {},
    total: e.emissions_tco2e,
    percent_of_scope: e.percentage_of_total,
    data_quality: 80,
  }));

  return (
    <Box>
      {/* Header with stat card and add button */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Grid container spacing={2} sx={{ maxWidth: 600 }}>
          <Grid item xs={12} sm={6}>
            <StatCard
              title="Scope 1 Total"
              value={summary ? formatEmissions(summary.total_tco2e) : '--'}
              icon={<Factory sx={{ color: '#e53935' }} />}
              color="#e53935"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <StatCard
              title="Data Quality"
              value={summary?.data_quality_tier?.replace('_', ' ') || '--'}
              subtitle={`${categories.length} categories`}
            />
          </Grid>
        </Grid>
        <Button variant="contained" startIcon={<Add />}>
          Add Data
        </Button>
      </Box>

      {/* Source breakdown chart */}
      <Box sx={{ mb: 3 }}>
        <SourceBreakdown categories={categories} />
      </Box>

      {/* Gas breakdown */}
      {summary?.gas_breakdown && (
        <Box sx={{ mb: 3 }}>
          <GasBreakdown gasBreakdown={summary.gas_breakdown} />
        </Box>
      )}

      {/* Facility table */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Emissions by Facility / Entity
        </Typography>
        <FacilityTable facilities={facilities} />
      </Box>
    </Box>
  );
};

export default Scope1Page;
